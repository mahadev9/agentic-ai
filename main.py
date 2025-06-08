from uuid import uuid4

from graph import SimpleAgent

from dotenv import load_dotenv


load_dotenv(override=True)


def main():
    agent = SimpleAgent()
    thread_id = str(uuid4())

    while True:
        try:
            user_prompt = input("\nYou: ").strip()

            if user_prompt.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            if user_prompt.lower() == "tools":
                print("Available tools:")
                for tool in agent.get_available_tools():
                    print(f"- {tool['name']}: {tool['description']}")
                continue

            if not user_prompt:
                print("Please enter a valid prompt.")
                continue

            print("Agent: ", end="", flush=True)
            response = agent.chat(user_prompt, thread_id=thread_id)
            print(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
