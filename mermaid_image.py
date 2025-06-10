from graph import SimpleAgent

app = SimpleAgent()
app.agent.get_graph().draw_mermaid_png(output_file_path="workflow.png")
