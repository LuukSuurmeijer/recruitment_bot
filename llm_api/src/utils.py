from typing import Dict, Tuple, List
from transformers import AutoTokenizer
import outlines
import llama_cpp
import time
import yaml


def import_prompt(fp: str) -> Tuple[str, str]:
    """
    Read instruction prompt and possible default texts from a prompt.yaml file

    :param fp: Filepath to prompt file
    :return: Formatted chat template as a string
    """
    with open("prompt.yaml") as stream:
        contents = yaml.safe_load(stream)

    return contents["prompt"], contents["how_to_default"]


def prepare_chat_template(
    tokenizer: AutoTokenizer , chat: List[Dict[str, str]], system_prompt: str
) -> str:
    """
    Prepare a chat history for input to an LLM. I.E. Prepend system prompt and tokenize.

    :param tokenize: Tokenizer object of the model
    :param chat: Chat history with standard role, content format
    :param system_prompt: Instructions for the LLM
    :return: Formatted chat template as a string
    """
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
    Load a llama.cpp model from Huggingface into CPU. Assumes model is made using llama.cpp.

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
