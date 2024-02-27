# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.openai import OpenAI
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

load_dotenv('.env')

import os

from langchain_community.vectorstores.chroma import Chroma

"""
GLOBALS
"""
DEST_PATH = os.environ.get('DEST_PATH')
MODEL_NAME = os.environ.get("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
BASE_URL = os.environ.get('BASE_URL')


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


if __name__ == '__main__':
    print('RAG')

    # Define embedding function
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=BASE_URL)
    vector = Chroma(persist_directory=DEST_PATH, embedding_function=embedding_function)

    # Define a retriever interface
    retriever = vector.as_retriever()

    # Define LLM
    llm = OpenAI(
        model_name=MODEL_NAME,
        temperature=0,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Create a retrieval chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True,
                                           verbose=True)

    # Interact with RAG
    # query = "How many bytes are in the gold_fact_communications_messages_log table?"
    # query = "What is the population of Houston TX?"
    # query = "What is the population of New York City?"
    # query = "Compare the population of New York and Houston"
    query = "Compare the population of New York and Houston.  What is the percentage difference between the two populations?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
