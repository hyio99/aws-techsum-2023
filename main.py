import streamlit as st
import boto3
import json
from queryLLM import query_llm
from queryEmbeddings import inference_embeddings

# Define the API endpoint for the chatbot
endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-7b-f"
client = boto3.client("runtime.sagemaker", region_name="ap-southeast-2")

st.title('RAG with ChromaDB + Llama-2')
uploaded_image = st.file_uploader("Upload a document", type=["pdf", "docx"])
user_input = st.text_input('Enter text:', '')


if st.button('Query'):
    # Get embeddings 
    # query_embeddings = inference_embeddings(user_input)
    
    # TODO: perform similarity search w/ VectorDB
    # calling a lambda which performs that search, user will code

    # Query LLM with query + context
    # response = query_llm(user_input, context)

    response = query_llm(user_input)
    result = json.loads(response["Body"].read().decode())
    output = result[0]["generation"]["content"]
    st.text_area('Generated Text:', output, height=600)

