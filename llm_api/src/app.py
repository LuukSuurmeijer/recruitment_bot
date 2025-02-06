from models import LLMResponse
import outlines
import time
import os
import subprocess
from flask import Flask, request
from transformers import AutoTokenizer
from dotenv import load_dotenv

from utils import load_llamacpp_model, prepare_chat_template, import_prompt
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
SEED = 789001
DEFAULT_GENERATOR = outlines.generate.json(MODEL, LLMResponse)
tokenizer = MODEL.tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("HF_TOKENIZER"))

PROMPT, how_to_apply_text = import_prompt("prompt.yaml")
logger.info(PROMPT)


@app.route("/")
def hello_world():
    return "Not Implemented"


@app.route("/health", methods=["GET"])
def root():
    return "Healthy"


@app.route("/answer_question", methods=["POST"])
def answer_question():
    """
    This is the endpoint to the loaded LLM, it returns a new response based on the chat history and the system prompt
    ---
    tags:
      - [COMPANY NAME] Recruitment Bot Main Endpoint
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
      - name: output_len
        in: query
        type: int
        description: Desired number of output tokens
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
          default_response:
            type: bool
            description: Whether the answer was LLM-generated or a default text

    """

    # Parse request  body
    input = request.get_json()
    chat_history = input.get("chat")
    sampling_args = input.get("sampling_args")
    output_len = input.get("max_len", 2096)

    # Create the sampler if the request body contains alternative sampling parameters
    if sampling_args:
        logger.info("Got non-default sampling args, building generator.")
        generator = outlines.generate.json(
            MODEL,
            LLMResponse,
            sampler=outlines.samplers.multinomial(**sampling_args[0]),
        )
    else:
        generator = DEFAULT_GENERATOR

    # Prepare chat template for LLM
    logger.info("Preparing Chat Template")
    prompt = prepare_chat_template(TOKENIZER, chat_history, PROMPT)
    prompt_tokens = len(TOKENIZER.encode(prompt))

    # Generate the response
    logger.info("Generating Response")
    init_time = time.time()
    sequence = generator(prompt, max_tokens=prompt_tokens + output_len, seed=SEED)
    gen_time = time.time() - init_time
    tokens = len(tokenizer.encode(str(sequence))[0])
    tps = tokens / gen_time
    elapsed = time.time() - init_time

    logger.info(f"Generated {tokens} tokens in {elapsed} seconds")
    logger.info(f"{sequence}")
    # Prepare response body
    llm_response = sequence.model_dump()
    response_template = {
        "time": elapsed,
        "tokens": tokens,
        "tps": tps,
    }

    # If the LLM tagged the query as "application", return default text as answer
    if llm_response["topic"] == "application":
        print(how_to_apply_text)
        llm_response["answer"] = how_to_apply_text
        response_template["default_response"] = True
    else:
        response_template["default_response"] = False

    response = response_template | llm_response  # cool syntax for merging dicts

    logger.info(response)

    return response


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port)
