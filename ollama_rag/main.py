import pysqlite3
import sys

from langchain_core.prompts import ChatPromptTemplate

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field


from dotenv import load_dotenv
from langchain.chains import RetrievalQA, LLMMathChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

load_dotenv('.env')

import os

from langchain_community.vectorstores.chroma import Chroma

"""
GLOBALS
"""
MODEL_NAME = os.environ.get("MODEL_NAME")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


if __name__ == '__main__':
    print('RAG')
    # load from disk
    # query = 'What does Limeade use for a message bus'
    # query = "Who was the fastest random speaker in the world"
    # query = "Who led the league in scrimmage yards"
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    vector = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    # Define a retriever interface
    retriever = vector.as_retriever()

    # Define LLM
    llm = Ollama(model="mistral",
                 temperature=0,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

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

    while True:
        query = input("User: ")
        llm_response = qa_chain(query)
        print(f'Agent: {llm_response["result"]}') # 119152398