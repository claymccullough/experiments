# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA, LLMMathChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.openai import OpenAI
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool

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
    llm_math = LLMMathChain.from_llm(llm, verbose=True)

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
                                           # return_source_documents=True,
                                           verbose=True)

    tools = [
        Tool(
            name="dbt project",
            func=qa_chain.run,
            description="useful for when you need to answer questions about the documents in the database"
        ),
        Tool(
            name="llm-math",
            func=llm_math.run,
            description="Useful for when you need to answer questions about math."
        )
    ]

    agent = initialize_agent(tools, llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # query = """Who is the owner of the DIM_CUSTOMERS_V1 model?"""
    # query = """When was the DIM_CUSTOMERS_V1 model created?"""
    # query = "How many bytes is the `model.jaffle_shop.fct_orders` table consuming?  Also give me the answer in KB"
    # query = "List all the models that have the FIRST_ORDER column. Also, what is the type of that column in each table respectively?" # BREAKS IT
    # query = "List out all the tables in the dbt project with 'raw' in the name"
    # query = "Sum the total bytes consumed by all the models in the dbt project. Also give me the answer in GB." # DOESNT DO WELL WITH AGGREGATION - may want to present it to the LLM up front.
    query = "List out the columns in the `dim_customers.v2` model with their data types."
    agent.run(query)
