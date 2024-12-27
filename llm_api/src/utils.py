from typing import Dict, Tuple
import outlines
import llama_cpp
import time


def prepare_chat_template(
    tokenizer, chat: Dict[str, str], system_prompt: str
) -> Dict[str, str]:
    system_turn = {"role": "system", "content": system_prompt}
    if chat is None:
        chat = system_turn

    else:
        chat.insert(0, system_turn)
    return tokenizer.apply_chat_template(chat, tokenize=False)


def load_llamacpp_model(
    repo: str, modelfile: str, tokenizer: str
) -> Tuple[outlines.models.LlamaCpp, float]:
    """
    Load a llamacpp model from Huggingface into CPU. Assumes model is made using llama.cpp.

    :param repo: Repo string to Huggingface
    :param modelfile: Which file from the repo to download (or retrieve from cache)
    :param tokenizer: Path to Huggingface Tokenizer
    :return: Tuple of model and elapsed time for loading

    """
    t0 = time.time()
    model = outlines.models.llamacpp(
        repo,
        modelfile,
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(tokenizer),
    )
    t1 = time.time() - t0
    return model, t1
