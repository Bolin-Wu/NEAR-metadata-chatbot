from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

st.title("My First Metadata Chatbot Prototype")
st.write("Hello! This will soon search our database metadata.")

name = st.text_input("Your name?")
if name:
    st.write(f"Welcome, {name}! Let's build something cool.")

uploaded_file = st.file_uploader("Upload an Excel metadata file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Preview of your metadata:")
    st.dataframe(df.head(10))  # Shows first 10 rows
    
    
# if using mica-obiba:
# from obiba.mica import MicaClient

# client = MicaClient.build('https://www.maelstrom-research.org')
# client.authenticate(username='your_user', password='your_pass')

# # Example: search variables/studies
# result = client.get('/ws/variables/_search', params={'query': 'your_filter', 'limit': 1000})
# variables = result.json()  # Process into LangChain Documents