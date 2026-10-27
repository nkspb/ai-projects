import sys
import ollama

model = "qwen2.5:7b"
messages = [
    { "role": "system", "content": "You are a helpful IT engineer" },
]
model_options = {
    "num_ctx": 1024,
    "num_predict": 256
}

last_prompt_tokens = 0 
REPLY_RESERVE = 256
MAX_TOKENS_THRESHOLD = model_options["num_ctx"] - REPLY_RESERVE

try:
    while True:
        # Request user input
        user_message = input("\nEnter your message: ").strip()

        if not user_message: continue # Skip empty user message

        if user_message == "/exit":
            print("Quitting the app...")
            sys.exit()
        elif user_message == "/clear":
            messages = messages[:1] 
            print("History cleared")
            continue
        elif user_message == "/tokens":
            print(f"Last prompt_eval_count: {last_prompt_tokens}")
            continue

        # Build a request separately
        # The history is commited only after a successful response
        request_messages = messages + [
            { "role": "user", "content": user_message}
        ]
        

        # Estimate size of entire messages list AFTER adding last user message
        estimated_tokens = sum(len(message["content"]) for message in request_messages) / 4
            

        while estimated_tokens > MAX_TOKENS_THRESHOLD and len(request_messages) > 2:
            del request_messages[1:3]
            estimated_tokens = sum(len(message["content"]) for message in request_messages) / 4

        # Handling an edge case when the last two messages still exceed the budget:
        if estimated_tokens > MAX_TOKENS_THRESHOLD:
            print("Your message is too long. Please shorten it.")
            continue # Ask for input again
        try:
            print("Making request to Ollama...")
            # Send request to LLM
            stream = ollama.chat(
                model=model,
                messages=request_messages,
                options = model_options,
                stream=True
                )
            # Collect LLM reply and append it to messages
            llm_reply = ""
            prompt_tokens = 0
            reply_finished = False

            for chunk in stream:
                chunk_message = chunk.message.content 
                print(chunk_message, end="", flush=True)
                llm_reply += chunk_message

                if chunk.done:
                    prompt_tokens = chunk.prompt_eval_count 
                    reply_finished = True

            if not reply_finished:
                raise RuntimeError("The response ended before completion.")
        except Exception as error:
            print(f"\nRequest failed: {error}")
            continue       

        request_messages.append({"role": "assistant", "content": llm_reply})
        messages = request_messages
        last_prompt_tokens = prompt_tokens 

except (KeyboardInterrupt, EOFError):
    print("\nQuitting the app...")
