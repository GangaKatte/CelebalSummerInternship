import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------ Load Dataset ------------------
@st.cache_data
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.fillna("unknown", inplace=True)
    df['combined'] = df.apply(lambda row: " | ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
    return df

# ------------------ Embedding Model ------------------
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ Text Generation Model ------------------
@st.cache_resource
def load_smart_model():
    model_id = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

# ------------------ Answer Generator ------------------
def generate_answer(question, context, gen_model):
    prompt = f"""
You are a helpful assistant answering questions about loan approvals based on the following dataset context.

Context:
{context}

Question: {question}
Answer:
"""
    result = gen_model(prompt)
    return result[0]["generated_text"].split("Answer:")[-1].strip()

# ------------------ Search Context ------------------
def get_top_k_context(question, embeddings, texts, embedder, k=3):
    question_embedding = embedder.encode([question])
    sims = cosine_similarity(question_embedding, embeddings)[0]
    top_k_idx = sims.argsort()[-k:][::-1]
    top_context = [texts[i] for i in top_k_idx]
    return "\n".join(top_context)

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="Loan Approval RAG Chatbot", layout="wide")
st.title("🤖 RAG Q&A Chatbot – Loan Approval Dataset")
st.markdown("Ask anything about the loan approval dataset. Example: *What affects loan approval?*")

# Load data
csv_file = "loan_data.csv"
if not os.path.exists(csv_file):
    st.error("❌ Dataset file 'loan_data.csv' not found in current directory.")
    st.stop()

df = load_dataset(csv_file)
texts = df["combined"].tolist()
embedder = get_embedding_model()
embeddings = embedder.encode(texts, show_progress_bar=True)

# Load text generation model
gen_model = load_smart_model()

# User input
question = st.text_input("🔎 Your Question:", placeholder="Does credit history impact loan approval?")
if question:
    context = get_top_k_context(question, embeddings, texts, embedder)
    answer = generate_answer(question, context, gen_model)

    st.markdown("### ✅ Answer")
    st.success(answer)

    with st.expander("📂 Context Used"):
        st.write(context)
