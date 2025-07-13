import streamlit as st
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
DATA_PATH = "loan_data.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
TEXTS_PATH = "texts.pkl"

#Dataset
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    df.fillna("unknown", inplace=True)
    df['combined'] = df.apply(lambda row: (
        f"{row['Gender']} applicant, Married: {row['Married']}, "
        f"Education: {row['Education']}, Self-employed: {row['Self_Employed']}, "
        f"Applicant Income: {row['ApplicantIncome']}, Coapplicant Income: {row['CoapplicantIncome']}, "
        f"Loan Amount: {row['LoanAmount']}, Credit History: {row['Credit_History']}, "
        f"Loan Status: {'Approved' if row['Loan_Status'] == 'Y' else 'Rejected'}."
    ), axis=1)
    return df, df["combined"].tolist()


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_generator():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

def compute_or_load_embeddings(texts, embedder):
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(TEXTS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f1, open(TEXTS_PATH, "rb") as f2:
            return pickle.load(f1), pickle.load(f2)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    with open(EMBEDDINGS_PATH, "wb") as f1, open(TEXTS_PATH, "wb") as f2:
        pickle.dump(embeddings, f1)
        pickle.dump(texts, f2)
    return embeddings, texts

def get_top_k_context(question, embeddings, texts, embedder, k=3):
    question_embedding = embedder.encode([question])
    sims = cosine_similarity(question_embedding, embeddings)[0]
    top_k_idx = sims.argsort()[-k:][::-1]
    return "\n\n".join([texts[i] for i in top_k_idx])

def generate_answer(question, context, gen_model):
    prompt = f"""You are an expert loan approval assistant.
Based on the following applicant data, answer the question with insight and clarity.

Applicant Data:
{context}

Question: {question}
Answer:"""
    result = gen_model(prompt)[0]["generated_text"]
    return result.strip()

#UI
st.set_page_config("🤖 Fast Loan Approval Chatbot", layout="centered")
st.title("🤖 Fast Loan Approval Q&A Chatbot")
st.markdown("Ask about the loan dataset. *Example: How does credit history affect loan approval?*")


if not os.path.exists(DATA_PATH):
    st.error(f"❌ '{DATA_PATH}' not found. Please download the loan dataset and rename it as 'loan_data.csv'.")
    st.stop()

with st.spinner("📊 Loading dataset..."):
    df, readable_texts = load_dataset(DATA_PATH)

# enbedding
embedder = load_embedder()
with st.spinner("🔎 Generating or loading embeddings..."):
    embeddings, texts = compute_or_load_embeddings(readable_texts, embedder)


generator = load_generator()

#Q&A
question = st.text_input("🔍 Ask a question:", placeholder="e.g. Does coapplicant income impact approval?")
if question:
    with st.spinner("💬 Thinking..."):
        context = get_top_k_context(question, embeddings, texts, embedder)
        answer = generate_answer(question, context, generator)

    st.markdown("### ✅ Answer")
    st.success(answer)

    with st.expander("📂 Context Used"):
        st.text(context)
