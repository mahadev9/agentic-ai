import os
from tools import get_current_time, calculator, weather_tool, web_search_tool
from typing import List, Optional, TypedDict, Annotated, Dict, Any
import operator
import sqlite3

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_action: Optional[str]


class SimpleAgent:

    def __init__(self):
        self.model_name = "gemini-2.5-flash-preview-05-20"
        self.tools = [
            get_current_time,
            calculator,
            weather_tool,
            web_search_tool,
        ]
        self.llm = self._setup_llm()

        os.makedirs("./db", exist_ok=True)
        conn = sqlite3.connect("./db/agent_state.db", check_same_thread=False)
        self.memory = SqliteSaver(conn)
        self.agent = self._create_graph()

    def _setup_llm(self):
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        return llm.bind_tools(self.tools)

    def _create_graph(self):
        tool_node = ToolNode(self.tools)

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent", self._should_continue, {"tools": "tools", "end": END}
        )

        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.memory)

    def _should_continue(self, state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "end"

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]

        system_message = """You are a helpful AI assistant with access to multiple tools:

1. **google_search**: Search the web for current information
2. **get_weather**: Get weather information for any location
3. **calculator**: Perform mathematical calculations
4. **get_current_time**: Get current date/time in any timezone

Use these tools when needed to provide accurate, helpful responses. Always explain your reasoning and provide context for the information you find."""

        if not messages or not isinstance(messages[0], HumanMessage):
            messages = [HumanMessage(content=system_message)] + messages

        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def chat(self, message: str, thread_id: str = "default") -> str:
        try:
            # Create initial state
            initial_state = {"messages": [HumanMessage(content=message)]}

            # Run the graph
            config = {"configurable": {"thread_id": thread_id}}
            result = self.agent.invoke(initial_state, config)

            # Get the last message from the agent
            final_messages = result["messages"]
            agent_messages = [
                msg for msg in final_messages if isinstance(msg, AIMessage)
            ]

            if agent_messages:
                return agent_messages[-1].content
            else:
                return "I apologize, but I couldn't process your request properly."

        except Exception as e:
            return f"Error: {str(e)}"

    def get_available_tools(self) -> List[str]:
        tool_info = []
        for tool in self.tools:
            tool_info.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "args": str(tool.args) if hasattr(tool, "args") else "No args info",
                }
            )
        return tool_info
