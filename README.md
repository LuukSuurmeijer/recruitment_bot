# Chatbot for Recruitment Queries

I created this Chatbot as part of the application process for the GenAI team at Schiphol.

## Usage

The bot consists of a frontend and a backend API. The chatbot can be run in two ways.
- Spin up the frontend and backend individually directly on your machine
    - for the frontend, run `streamlit run frontend/src/frontend.py`
    - for the backend, run `flask run host='0.0.0.0'` from the llm_api/src directory
- Run it in docker
    - run `docker compose up` to build and run the containers
    - Note that for demo purposes, both containers use the host network to communicate with each other.

After running the components, you should have a frontend running on localhost:8501 and a flask api running on localhost:5000
You can chat to the bot via the frontend, or make API requests to localhost:5000/answer_question. The endpoint is documented in the code.

Note that running the code requires about ~15GB of memory and a modelfile of roughly 2.5GB will be downloaded from Huggingface.

You can modify the instruction prompt for the LLM by modifying the `prompt:` field in `llm_api/src/prompt.yaml`.

## Process and priorities.

Going into this assignment, I had a number of priorities. I wanted to try and host my own LLM to showcase flexibility beyond querying the OpenAI API. I secondly wanted to include a feature that would set it apart from a simple wrapper around an LLM call. I really like Structured Generation so decided to incorporate that (see below). I additionally wanted to include the possibility of answering with static text rather than an LLM response, in the case of for exmaple common questions FAQ questions. Due to the time constraints I just implemented one such static intent, and used structured generation to detect it (more on that below). Laslty, I wanted to build a somewhat robust software solution that is containerized and implemented in a flexibly parametrized and maintainable manner to which it would be relatively straightforward to add more components (such as a vector store for RAG). This is why I opted for docker, dot_env and proper logging.

- For the frontend, I used streamlit for a basic chat UI. I am no front-end engineer, and streamlit is easy enough to use in python and looks decent. Additionally it comes with the ability to track session state, which is useful for saving conversations for evaluation.
- For the backend I used a Flask application. FastAPI is the more standard approach these days, but I have more experience using Flask and wanted to save time.

### Model

For the LLM I used the biggest version of Llama 3.2 I could run, which turned out to be Llama3.2 3B. I used the instruct-tuned version of the model as this is a chatbot application. I additionally used a 6bit-quantized version in gguf format from Huggingface. This allowed me to run the model on my CPU using a llama.cpp backend. The model is surprisingly good for its size, although it does sometimes generate gibberish or unhelpful answers.


## Features

### Structured Generation

The Chatbot makes use of Structured Generation via the outlines package. This means that the LLM output can be guided to adhere to some desired syntax.
The LLM will respond with a Pydantic object (LLMResponse) and fill out the fields defined, which is converted to JSON.
If you're interested: This works by applying a finite-state-automaton implementing the desired grammar to the output probabilities, and setting the probability mass of unreachable states to 0. This guarantees adherence to the specified grammar (except when the LLM generates syntactical tokens in the free parts of the response).
Using Structured Generation: 
    - Reduces hallucination and improves reliability.
    - Allows you to use the LLM for multiple purposes at the same time, in this case classifying queries.
    - Makes processing of the LLM response much easier.
This makes this feature highly interesting for such a use-case!

### Static answers

I imagined recruiters receive a bunch of questions that can be found on the website, so I wanted to include the possibility of more traditional chatbot style answers like the FAQ. This requires classification of use intents. Although in a production setting I would prefer using an intent and classification model like RASA, this is also possible with structured generation. This was easier to implement under the time constraints. With structured generation I instructed the model to classify the intent of the user according to a number of options defined in `models.py`, so that for specific intens a default text can be returned. The intent I added is the "how can I apply intent" and the default text is added to `prompt.yaml`. I took the text from the Schiphol vacancy website FAQ. This way the model can be used in a more reliable way if there should be a standard answer.

## RAG

For the chatbot to be useful, a RAG component is basically necessary. I would implement this as follows:

- Add a vector store as a component to the stack. I've used [Milvus](https://milvus.io/) before, so I would probably use it here too. It has functionality for chunking, fast vector comparison and reranking.
- The documents to be added would be mainly the current listed vacancies at Schiphol.
- As for the embeddings, I would probably want to use some encoder-only transformer such as Roberta. Recently ModernBERT was released, which has a much longer context window making it more suitable for retrieval. I would not resort to simpler models, as they are often worse than a BM25 baseline. I would evaluate comparing to BM25.
- I would opt for paragraph-based chunking as a baseline. Milvus associates each chunk with the source document so the entire document is never lost. A possible improvement could be to chunk based on a more semantic structure of the vacancies. A job ad usually has a fixed structure of "Who are we", "what will you do", "what are your qualifications" so I would be curious to experiment with chunking based on that strucutre.

The RAG pipeline would then look like
- Before each query to the LLM. We search Milvus for related chunks, rerank them and incorporate them into the prompt. The LLM would then generate a response based on that. I would make sure to return links to the used sources and quote the relevant passages where the answer might be.

## Evaluation

I think ultimately the success of the chatbot has to measured with respect to some KPI's relevant to the staff it is supposed to help. The primary goal is to reduce the number of calls / queries Schiphol's recruiters get, so we would have to start measuring non-chatbot call / query volume. To further evaluate whether the bot is helpful, I would include UI elements that allow the user to give feedback to the bot responses. These feedback signlas combined with the fact this bot already classifies queries makes it so you can do useful analyses about which types of queries are helpful and which are not. Sessions would also have to be saved to some sort of database so that they can be analyzed (provided they pass the privacy checks, see below).

I would also evaluate the retrieval component seperately, as the quality of the retriever affects the LLM output greatly. I would do this by creating some sort of static, simple, custom eval dataset with Schiphol's vacancies and documents. The dataset would consist of some common queries and a known set of documents that should be retrieved for that query. I think even a dataset of ~30 question - document pairs could be enough to get some reliable estimate of how well the retriever is performing. Metrics I would compute are F-score, Precision at k (for however many documents the system is set to retrieve) and mean average precision.

I would test the generator component seperately by A / B testing prompts and seeing the effect on the user feedback.

## Privacy

Privacy is tricky. I think this problem has two sides:

1. Private information submitted by the user
2. Private information about Schiphol and its employees outputted by the chatbot.

The first problem is one of the reasons I would consider using an intent/entity classification model like RASA. RASA has both deep learning as well as symbolic (regex etc.) functionality for detecting certain types of information such as addresses, contact information and banking information. If information like this is detected, the user should be prompted to not include it in the query and the LLM should not answer (this is why it is important to have functionality for static answers). The query and response should not be saved for evaluation.

The second problem would have to be tackled by evaluating which documents contain sensitive information (possibly using a similar approach as above). These should not be ingested into the vector store. As a final check we could run the same private-information detection steps that I would apply to the user input to validate whether the LLM response doesn't contain any private information.

## Improvements and Ideas

Many of these have already been mentioned in the rest of the text.

1. As mentioned I would add an intent/entity classification module in the form of RASA. This would enable a ton of useful features for the bot.
2. A database where sessions are saved for analysis and evaluation.
3. UI elements allowing for user feedback.
4. Experiment with larger / better models
5. More sophisticated Structured Generation grammars
6. Structured way to A / B test prompts. A prompt store?

## Towards Pilot & Deployment

To make this into a deployable pilot, what minimially has to be implemented is the RAG component. If this is implemented I would add some more developer best practices like

- Unit tests
- Integration tests for static answers
- Some sort of CI/CD pipeline
- Get it running on Azure / AWS

Then it has to be integrated with the website. I would assume there are other people working on the website, so we would have to consult the developers and UX designers on how to best incorporate the bot.

Afterwards we could work on private information detecetion, implementing RASA, better UI, etc.


## Time spent