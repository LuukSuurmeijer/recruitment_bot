import streamlit as st
import requests
import os
from dotenv import load_dotenv
import logging

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger = logging.getLogger()
logging.basicConfig(
    format=" %(name)s :: %(levelname)-2s :: %(message)s", level=LOG_LEVEL
)

st.title("[COMPANY NAME] Recruitment Bot")

# Initialize chat history & response history for logging / saving
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.responses = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get LLM Response
    response = requests.post(
        url=f"{os.getenv('LLM_HOST')}{os.getenv('LLM_ENDPOINT')}",
        json={"chat": st.session_state.messages},
    ).json()

    logging.info("Got LLM API Response")

    # Display LLM response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["answer"])

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"]}
    )
    st.session_state.responses.append(response)

    logging.info("Session state saved")

