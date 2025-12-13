import os
from dotenv import load_dotenv
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Define paths
#Note: os.path.dirname(__file__) is used to navigate relative to this file's location.
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
KB_PATH = os.path.join(DATA_DIR, 'knowledge_base')
ASSET_PATH = os.path.join(DATA_DIR, 'assets.json')

load_dotenv(os.path.join(os.path.dirname(__file__), '../../..', '.env'))
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME")
print("HuggingFace Mode: ", HF_MODEL_NAME)

def load_structured_assets():
    """Loads mock IT asset data from the JSON file."""
    print(f"Loading assets from: {ASSET_PATH}")
    if not os.path.exists(ASSET_PATH):
        print(f"ERROR: Asset file not found at {ASSET_PATH}")
        return []
    with open(ASSET_PATH, 'r') as f:
        return json.load(f)

def setup_chroma_db():
    """Sets up the ChromaDB RAG vector store using HuggingFace embeddings."""
    print(f"Loading knowledge base documents from: {KB_PATH}")
    
    #1) Load documents
    loader = DirectoryLoader(KB_PATH, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    
    #2) Split documents (Good practice for RAG)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    #3) Create HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, #'cuda' for Nvidia GPU, else 'cpu'
        encode_kwargs={'normalize_embeddings': False}
    ) #
    
    #4) Create Vector Store 
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    print(f"ChromaDB setup complete with {len(texts)} chunks using {HF_MODEL_NAME}.")
    return vectorstore

#example:
if __name__ == "__main__":
    
    assets = load_structured_assets()
    if assets:
        print(f"\nLoaded {len(assets)} assets. Example Asset ID: {assets[0]['Asset_ID']}")
    
    vector_db = setup_chroma_db()
    
    query = "Why is my hardware lost and do I contact?"
    #test retrieval
    results = vector_db.similarity_search(query, k=1)
    print("\n--- RAG Test Result ---")
    print(f"Query: {query}")
    print(results[0].page_content)