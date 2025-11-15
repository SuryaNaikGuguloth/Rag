import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="SystemVerilog Chatbot", layout="wide")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("hf")
os.environ["GOOGLE_API_KEY"] = os.getenv("googleapikey")

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

#torch.set_default_device("cpu")

with open("pdf_extracted_text.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)
texts = text_splitter.split_text(text_data)


embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

index = FAISS.from_texts(texts, embedding=embed_model)

st.title("🤖 SystemVerilog Documentation Chatbot")
st.markdown("Ask anything about SystemVerilog Verification based on your document!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Enter your SystemVerilog question...")

if user_query:
    def retrieve(query, k=10):
        results = index.similarity_search(query, k=k)
        context = "\n\n".join([r.page_content for r in results])
        return context, results

    context, docs = retrieve(user_query)

    prompt=f"""
You are an expert **SystemVerilog Verification Engineer**, **technical author**, and **educator** with over 20 years of experience in digital design and verification.

Your output MUST be strictly grounded in the retrieved document context.

-----------------------------------------------------
📘 Retrieved Context (from RAG system):
{context}
-----------------------------------------------------

❓ User Query:
{user_query}

-----------------------------------------------------
🧠 STRICT TASK RULES (READ CAREFULLY)

1. PRIMARY REQUIREMENT — RAG-ONLY ANSWERING
   - **Use ONLY the retrieved context** for all technical content.
   - If the answer requires information not present in the context:
     → Respond with: **"Insufficient document context to answer."**
   - DO NOT use your own knowledge, memory, or assumptions.

2. CONTENT HANDLING
   - If context chunks break sentences or code, merge them logically.
   - Only reorganize, reformat, or clarify what already exists.
   - NO new concepts, NO new definitions, NO external SystemVerilog knowledge.

3. ALLOWED USE OF EXPERTISE
   (Only for improving existing context — NOT adding new content)
   - Reformat SystemVerilog code for readability.
   - Fix syntax errors in context code.
   - Complete truncated code **only when the missing part is implied**.
   - Reconstruct minimal code to match given output (if output appears in context).
   *Never add features or logic not traceable to the context.*

4. CODE & OUTPUT RULES
   - If code + output exist in context → present both exactly.
   - If only output exists → reconstruct minimal SystemVerilog code.
   - Label reconstructed code as: **(Reconstructed Based on Output Behavior)**
   - Show outputs in fenced ```Output``` blocks.

5. SECTION / SAMPLE IDENTIFICATION
   - If context includes Section, Example, Sample labels → include them.
   - If multiple → list all.
   - If none → skip.

6. FORMAT OF FINAL ANSWER (REQUIRED)
   Your response MUST follow this structure:

   **📄 Section Reference:** Section <no> / Sample <no> / Derived from Chunk <no(s)>

   **Document-Based Explanation**
   <Strictly derived from context>

   **Code Examples**
   ```systemverilog
   <Code from context or reconstructed>
"""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_query)
    ]

    response = chat.invoke(messages)

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": response.content})

for chat_msg in st.session_state.chat_history:
    if chat_msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(chat_msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(chat_msg["content"])
