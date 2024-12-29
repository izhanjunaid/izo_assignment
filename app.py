import streamlit as st
import requests
import time

# FastAPI Backend URL
BASE_URL = "http://localhost:8000"

# Health Check
def check_health():
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        return response.json()
    return {"status": "unhealthy", "message": "Unable to reach API"}

# Upload Document
def upload_document(file):
    files = {"file": file}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to upload document"}

# Query Documents
def query_documents(query_text):
    json_data = {"text": query_text}
    response = requests.post(f"{BASE_URL}/query", json=json_data)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to query documents"}

# Change Embedding Model
def change_embedding_model(model_name):
    json_data = {"model_name": model_name}
    response = requests.post(f"{BASE_URL}/change-model", json=json_data)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to change model"}

# Streamlit UI
st.title("Document QA System")

# Health Check
health_status = check_health()
st.write(f"Health Status: {health_status['status']}")

# Upload Document
st.header("Upload a Document")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
if uploaded_file:
    with st.spinner("Uploading document..."):
        result = upload_document(uploaded_file)
        if "message" in result:
            st.success(result["message"])
            st.write(f"Document ID: {result['document_id']}")
        else:
            st.error(result.get("error", "Unknown error"))

# Query Documents
st.header("Query the Document")
query_text = st.text_area("Enter your query:")
if st.button("Submit Query") and query_text:
    with st.spinner("Processing query..."):
        query_result = query_documents(query_text)
        if "response" in query_result:
            st.write("Query Response:")
            st.write(query_result["response"])
            st.write(f"Confidence Score: {query_result['confidence_score']}")
            st.write(f"Processing Time: {query_result['processing_time']:.2f} seconds")
        else:
            st.error(query_result.get("error", "Unknown error"))

# Change Embedding Model
st.header("Change Embedding Model")
embedding_model = st.selectbox("Select Model", ["all-mpnet-base-v2", "all-MiniLM-L6-v2"])
if st.button("Change Model"):
    with st.spinner("Changing embedding model..."):
        model_result = change_embedding_model(embedding_model)
        if "message" in model_result:
            st.success(model_result["message"])
        else:
            st.error(model_result.get("error", "Unknown error"))

