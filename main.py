#!/usr/bin/env python3

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from langchain.llms import LlamaCpp
import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def __enter__(self):
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")


class Llama:
    def __init__(
        self,
        model_path="models/llama-2-7b-chat.ggmlv3.q4_K_S.bin",
        debug=False,
        n_gpu_layers=1,
        n_batch=512,
        callback_manager=None,
    ):
        self.debug = debug
        # https://github.com/abetlen/llama-cpp-python/blob/92c077136d1f0b029f8907a79eae009a750005e2/llama_cpp/llama.py#L42
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=True,  # MUST set to True
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=self.debug,
        )

    def run(self, prompt):
        if self.debug:
            with Timer():
                self.llm(prompt)
        else:
            self.llm(prompt)


def main():
    print("Running...")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    verbose = False
    llama = Llama(debug=verbose, callback_manager=callback_manager)

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
