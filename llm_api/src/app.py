from models import LLMResponse
import outlines
import time
import os
import subprocess
from flask import Flask, request
from transformers import AutoTokenizer
from dotenv import load_dotenv

from utils import load_llamacpp_model, prepare_chat_template
import logging

load_dotenv()
app = Flask(__name__)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger = logging.getLogger()
logging.basicConfig(
    format=" %(name)s :: %(levelname)-2s :: %(message)s", level=LOG_LEVEL
)

IS_CHAT_MODEL = True
HF_REPO = os.getenv("HF_REPO")
HF_MODEL = os.getenv("HF_MODEL")
TOKENIZER_STRING = os.getenv("HF_TOKENIZER")

subprocess.run(["huggingface-cli", "login", "--token", os.getenv("HUGGINGFACE_TOKEN")])

logger.info("Loading model...")
MODEL, t = load_llamacpp_model(HF_REPO, HF_MODEL, TOKENIZER_STRING)
logger.info(f"Loaded model in {t} seconds")

# Construct structured sequence generator
DEFAULT_GENERATOR = outlines.generate.json(MODEL, LLMResponse)
tokenizer = MODEL.tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("HF_TOKENIZER"))

PROMPT = """Instruct: You are a Recruiter for an airport. You have seen thousands of job ads and candidates, and you know exactly what the airports needs from potential candidates. You answer queries from potential candidates. You must also tag queries with a topic. Content queries are related to the duties of a particular job. Practical queries are related to how you can apply. Personal queries are related to a candidates qualification and situation.
            Please return a JSON object with the answer to the query and topic of the query. Be as helpful as possible.
            This is the conversation so far:\n
        """


@app.route("/")
def hello_world():
    return "Not Implemented"


@app.route("/health", methods=["GET"])
def root():
    return "Healthy"


@app.route("/answer_question", methods=["POST"])
def answer_question():
    """
    This is the endpoint that the loaded LLM, it returns a new response based on the chat history and the system prompt
    ---
    tags:
      - Schiphol Recruitment Bot Main Endpoint
    parameters:
      - name: chat
        in: query
        type: List[Dict[str, str]]
        required: true
        description: The chat history of the session in standard OpenAI/HF format
      - name: sampling_args
        in: query
        type: Array
        description: Sampling parameters
            - parameters:
                - name: temperature
                - type: float
                - name: top_k
                - type: int
                - name: top_p
                - type: float

    responses:
      200:
        description: Output LLM Response
        schema:
          answer:
            type: string
            description: LLM Answer
          topic:
            type: string
            description: LLM Query Classification
          time:
            type: float
            description: Time it took to generate
          tokens:
            type: int
            description: Number of tokens in output
          tps:
            type: float
            description: Tokens per second

    """
    input = request.get_json()
    chat_history = input.get("chat")
    sampling_args = input.get("sampling_args")

    if sampling_args:
        logger.info("Got non-default sampling args, building generator.")
        generator = outlines.generate.json(
            MODEL,
            LLMResponse,
            sampler=outlines.samplers.multinomial(**sampling_args[0]),
        )
    else:
        generator = DEFAULT_GENERATOR

    logger.info("Preparing Chat Template")
    prompt = prepare_chat_template(TOKENIZER, chat_history, PROMPT)
    prompt_tokens = len(TOKENIZER.encode(prompt))

    init_time = time.time()
    sequence = generator(prompt, max_tokens=prompt_tokens + 2096)
    gen_time = time.time() - init_time
    tokens = len(tokenizer.encode(str(sequence))[0])
    tps = tokens / gen_time
    elapsed = time.time() - init_time

    logger.info(f"Generated {tokens} tokens in {elapsed} seconds")

    response_template = {
        "time": elapsed,
        "tokens": tokens,
        "tps": tps,
    }

    response = response_template | sequence.model_dump()

    logger.info(response)

    return response


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))  # Default to 5000
    app.run(host="0.0.0.0", port=port)
