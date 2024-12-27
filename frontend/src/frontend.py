import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Schiphol Recruitment Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = requests.post(
        url=f"{os.getenv('LLM_HOST')}{os.getenv('LLM_ENDPOINT')}",
        json={"chat": st.session_state.messages},
    ).json()

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"]}
    )
