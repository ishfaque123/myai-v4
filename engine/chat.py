from core import chat, _memory_path


def main():
    print("Nivora AI")
    print("Type 'exit' to quit or 'clear' to clear this session.")

    session_id = "terminal"

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNivora AI: Goodbye!")
            break

        if not user:
            continue
        if user.lower() == "exit":
            print("Nivora AI: Goodbye!")
            break
        if user.lower() == "clear":
            path = _memory_path(session_id)
            if path.exists():
                path.unlink()
            print("Nivora AI: Memory clear kar di.")
            continue

        result = chat(user, session_id)
        print("Nivora AI:", result["reply"])


if __name__ == "__main__":
    main()
