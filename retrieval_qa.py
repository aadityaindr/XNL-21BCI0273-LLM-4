from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

def setup_retrieval_qa(vectordb):
    llm_pipeline = pipeline(
        'text-generation',
        model='EleutherAI/gpt-neo-125M',
        max_length=256,
        temperature=0.2,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    prompt_template = """
    You are a real-time financial assistant chatbot. Provide concise and accurate answers using the context provided.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain
