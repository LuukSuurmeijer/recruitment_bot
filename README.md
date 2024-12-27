# Chatbot for Recruitment Queries

## Model

## Usage

## Logic Priorities

I had a number of priorities when building this bot, mainly focussed around flexibility. I firstly wanted to be easily be able to swap out models and prompts. If a newer model comes out that performs better, is more well adjusted for Dutch, or runs more efficiently, it should be easy to swap it for the current model and restart the chatservice.

Secondly, it's important to keep testing prompts and change them as performance differences start to show or new business needs arise.

Thirdly, I wanted to be able to save a lot of information about the conversations and queries. This is relevant for evaluating how well the bot does as well as collecting information on the types of queries people ask the bot. This information can be used to further improve the solution, or even fine-tune the model specifically for Schiphol's use-case.

Lastly, I wanted to use some SOTA techniques in using LLMs like structured generation. This allows the model to hallucinate less (not in terms of content, but in terms of irrelevant information), enables guaranteed syntactically valid json in the output (extremely useful for downstream processing of the responses), and it allows you to have the LLM perform several tasks with a single query (like tagging the conversation with a topic label).

In order to be able to save and view data about a particular conversation with a customer, I added the Turn and Session logic.

### Structured Generation

    The Chatbot makes use of Structured Generation. This means that the LLM output can be guided to adhere to some desired syntax.
    The LLM will respond with a Pydantic object (LLMResponse) as described below.
    If you're interested: This works by applying a finite-state-automaton to the output probabilities, and setting the probability mass of unreachable states to 0.
    In this case, as a default, I opted to use simple JSON.
    Using Structured Generation: 
        - Reduces hallucination
        -

## UI

## RAG

## Privacy