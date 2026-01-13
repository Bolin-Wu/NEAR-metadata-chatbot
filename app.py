from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.title("My First Metadata Chatbot Prototype")
st.write("Hello! This will soon search our database metadata.")

name = st.text_input("Your name?")
if name:
    st.write(f"Welcome, {name}! Let's build something cool.")