from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def create_vector_db(text_chunks):
    vectordb = Chroma.from_texts(text_chunks, embedding_model)
    return vectordb
