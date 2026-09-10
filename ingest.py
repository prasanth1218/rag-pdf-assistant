from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "documents/sample.pdf"
INDEX_PATH = "faiss_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

print("Loading PDF...")
documents = PyPDFLoader(PDF_PATH).load()
print(f"  {len(documents)} pages")

print("Splitting...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)
chunks = splitter.split_documents(documents)
print(f"  {len(chunks)} chunks")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building index...")
vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
vectorstore.save_local(INDEX_PATH)

print(f"Done. {vectorstore.index.ntotal} vectors saved to {INDEX_PATH}/")