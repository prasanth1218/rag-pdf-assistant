import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------- STARTUP: runs once when the server boots ----------

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS index...")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

RAG_PROMPT = """You are a helpful assistant answering questions about a company handbook.

Answer the question using ONLY the context provided below.

Rules:
- Use only information found in the context. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "The information is not available in the provided document."
- Do not guess or invent details.
- Keep the answer clear and concise.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT)


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(f"Ready. {vectorstore.index.ntotal} vectors loaded.")

# ---------- REQUESTS: run on every call ----------

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    sources = [
        {"page": d.metadata.get("page_label"), "preview": d.page_content[:120]}
        for d in docs
    ]

    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, port=5000)