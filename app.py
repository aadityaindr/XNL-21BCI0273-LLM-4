import os
import streamlit as st
#import yfinance as yf
import PyPDF2

from utils import extract_text_from_image, extract_text_from_pdf, load_csv_data
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# ---------------- LLM Initialization ----------------
import yfinance as yf

try:
    stock = yf.Ticker("GOOG")
    data = stock.history(period="1d")
    print(data)
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
def initialize_llm():
    model_name = "EleutherAI/gpt-neo-125M"  # Free, open-source model
    generator = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=100,  # Number of tokens to generate
        temperature=0.0,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=generator)
    return llm


llm = initialize_llm()

# ---------------- Streamlit Title ----------------
st.title("📈 Real-Time Financial Assistant Chatbot")

# ---------------- Session State: Initialize History ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Sidebar: File Uploads ----------------
st.sidebar.header("Upload Multi-Modal Data")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")
uploaded_image = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type="csv")

text_data = ""

# Process uploaded PDF
if uploaded_pdf:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    text_data += pdf_text

# Process uploaded image via OCR
if uploaded_image:
    image_text = extract_text_from_image(uploaded_image)
    text_data += "\n" + image_text

# Process uploaded CSV data
if uploaded_csv:
    csv_text = load_csv_data(uploaded_csv)
    text_data += "\n" + csv_text

# If no files are uploaded, load a local document
if not text_data.strip():
    local_file_path = r"D:\genAI\data\document.pdf"
    if os.path.exists(local_file_path):
        with open(local_file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            document_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    document_text += page_text
        text_data = document_text
        st.sidebar.info(f"Loaded local document from: {local_file_path}")
    else:
        st.sidebar.warning("No uploaded files and no local file found at D:\\genAI\\data\\document.pdf.")

# ---------------- Utility Functions ----------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    return vectordb

def get_stock_price(ticker_symbol: str) -> str:
    try:
        ticker = yf.Ticker(ticker_symbol)
        price = ticker.fast_info.last_price
        if price:
            return f"{ticker_symbol.upper()} current price: ${price:.2f}"
        else:
            return f"Price data unavailable for {ticker_symbol.upper()}."
    except Exception as e:
        return f"Error fetching data for {ticker_symbol.upper()}: {e}"

# ---------------- RAG Setup ----------------
if text_data.strip():
    text_chunks = split_text(text_data)
    vectordb = create_vector_db(text_chunks)
    
    # Create a PromptTemplate instance for our QA chain
    prompt_template = """
You are a real-time financial assistant chatbot with multi-modal context retrieval capabilities.
You have access to live market data, historical financial reports, and relevant financial news.

Provide concise, non-repetitive answers. If uncertain, say so.

Context:
{context}

User: {question}

Financial Assistant:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Use chain_type "stuff" to allow a single prompt template
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
else:
    vectordb = None
    qa_chain = None

# ---------------- Main Chat Interface ----------------
user_query = st.text_input("Ask your financial query:")

if user_query:
    st.session_state.history.append(("You", user_query))
    
    if "stock price" in user_query.lower():
        words = user_query.split()
        possible_ticker = next((w for w in words if w.isupper() and len(w) <= 5), None)
        response = get_stock_price(possible_ticker) if possible_ticker else "Could not detect ticker symbol. Please specify one (e.g., 'AAPL')."
        source_documents = []
    elif qa_chain:
        try:
            response_dict = qa_chain({"query": user_query})
            response = response_dict.get("result", "No result found.")
            source_documents = response_dict.get("source_documents", [])
        except Exception as e:
            response = f"Error generating response: {e}"
            source_documents = []
    else:
        response = "Please upload data or ask about a stock price (e.g., 'stock price of AAPL')."
        source_documents = []
    
    st.session_state.history.append(("Bot", response))
    
    # Display source documents if available
    if source_documents:
        st.markdown("### Source Documents:")
        for doc in source_documents[:3]:
            st.markdown(f"- **Document:** {doc.metadata.get('source', 'Unknown')}")
            snippet = doc.page_content[:200].strip()
            st.markdown(f"  {snippet}...")

# Display conversation history (latest first)
for speaker, msg in reversed(st.session_state.history):
    st.markdown(f"**{speaker}:** {msg}")

# ---------------- Sidebar Disclaimer ----------------
st.sidebar.markdown("---")
st.sidebar.info("Disclaimer: This chatbot provides general financial information and is not a substitute for professional financial advice.")
