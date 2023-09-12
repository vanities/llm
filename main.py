#!/usr/bin/env python3

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from llama import Llama


def main():
    print("Running...")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    verbose = False

    models = {
        "code_llama": "codellama-34b-instruct.Q4_K_M.gguf",
        "70b_llama": "llama-2-70b-chat.ggmlv3.q4_K_M.bin",
    }
    current_model = models["code_llama"]
    llama = Llama(
        model_path=f"models/{current_model}",
        debug=verbose,
        callback_manager=callback_manager,
    )

    while True:
        try:
            prompt = input("\n\nEnter a Prompt (or 'exit' to quit): ")
            if prompt == "exit":
                break
            else:
                llama.run(prompt)
        except Exception as e:
            print(f"Exiting... {e}")


if __name__ == "__main__":
    main()
