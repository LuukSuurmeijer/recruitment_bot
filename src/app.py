from flask import Flask, request
import requests
import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

app = Flask(__name__)

PATH_TO_CONVERTED_WT = "/home/ubuntu/Llama-2-13b-chat-german/"
IS_CHAT_MODEL = True


print("Loading model...")
# TODO: what does wrapping in with `init_empty_weights():` do?
print(f"CUDA is available: {torch.cuda.is_available()}")

# TODO: test 'auto' instead of device to distribute over gpus
MODEL = LlamaForCausalLM.from_pretrained(
    PATH_TO_CONVERTED_WT, device_map="auto", torch_dtype=torch.bfloat16
)

TOKENIZER = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WT)


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/health", methods=["GET"])
def root():
    return "Healthy"


@app.route("/generate", methods=["POST"])
def answer_question():
    """Rewrite a text to simpler language using Llama.

    :param text: text to rewrite as str
    "return: dict with rewritten
    """
    input = request.get_json()
    prompt = input["prompt"]
    max_len = input.get("max_len", 50)

    start = time.time()
    # Encode prompt
    tokenized_inputs = TOKENIZER(prompt, return_tensors="pt")
    # Generate output
    generate_ids = MODEL.generate(
        tokenized_inputs.input_ids.to("cuda"),
        max_length=len(tokenized_inputs.input_ids[0]) + max_len,
    )
    # Decode output
    answer = TOKENIZER.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Cut prompt
    generated_sequence = answer[len(prompt) - 1 :]
    elapsed = time.time() - start

    print(f"generated sequences {generated_sequence} in {elapsed} seconds")

    response = {
        "output": generated_sequence,
        "time": elapsed,
        "tokens": len(generate_ids),
    }

    return response


if __name__ == "__main__":
    app.run()
