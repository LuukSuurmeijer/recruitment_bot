from models import *
import outlines
import llama_cpp
import time
import os
import subprocess
from flask import Flask, request
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

PATH_TO_CONVERTED_WT = "/home/ubuntu/Llama-2-13b-chat-german/"
IS_CHAT_MODEL = True


print("Loading model...")
subprocess.run(["huggingface-cli", "login", "--token", os.environ["HUGGINGFACE_TOKEN"]])
model = outlines.models.llamacpp(
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "Llama-3.2-3B-Instruct-Q4_K_S.gguf",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct"
    ),
)

# Construct structured sequence generator
generator = outlines.generate.json(model, LLMResponse)
tokenizer = model.tokenizer

# Draw a sample
SEED = 789005

PROMPT = "Instruct: You are a Recruiter for an airport. You have seen thousands of job ads and candidates, and you know exactly what the airports needs from potential candidates. You answer queries from potential candidates. You must also tag queries with a topic. Content queries are related to the duties of a particular job. Practical queries are related to how you can apply. Personal queries are related to a candidates qualification and situation. \nPlease return a JSON object with the answer to the query and topic of the query.\nOutput:"


@app.route("/")
def hello_world():
    return "Not Implemented"


@app.route("/health", methods=["GET"])
def root():
    return "Healthy"


@app.route("/answer_question", methods=["POST"])
def answer_question():
    """Rewrite a text to simpler language using Llama.

    :param text: text to rewrite as str
    "return: dict with rewritten
    """
    input = request.get_json()  # noqa: F821
    prompt = f"{PROMPT}{input['query']}"

    init_time = time.time()
    sequence = generator(prompt, seed=SEED, max_tokens=512)
    gen_time = time.time() - init_time
    tokens = len(tokenizer.encode(str(sequence))[0])
    tps = tokens / gen_time
    elapsed = time.time() - init_time

    response = {
        "time": elapsed,
        "tokens": tokens,
        "tps": tps,
    }

    return response | sequence.model_dump()


if __name__ == "__main__":
    app.run()
