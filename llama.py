from langchain.llms import LlamaCpp

from timer import Timer


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
