# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain.agents import initialize_agent, AgentType, create_structured_chat_agent
from langchain_core.tools import Tool


from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain, RetrievalQA, LLMMathChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv('.env')

import os

from langchain_community.vectorstores.chroma import Chroma

MODEL_NAME = os.environ.get("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL")
INFER_BASE_URL = os.environ.get("INFER_BASE_URL")


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
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=EMBED_BASE_URL)
    vector = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    # Define a retriever interface
    retriever = vector.as_retriever()

    # Define LLM
    llm = Ollama(model=MODEL_NAME,
                 base_url=INFER_BASE_URL,
                 temperature=0.9,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)

    # Create a retrieval chain to answer questions
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     # return_source_documents=True,
                                     verbose=True)

    tools = [
        Tool(
            name="General Knowledge",
            func=qa.run,
            description="useful for when you need to answer questions about the documents in the database"
        ),
        Tool(
            name="Llm-Math",
            func=llm_math.run,
            description="Useful for when you need to answer questions about math."
        )
    ]

    # Create agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    while True:
        # agent.run("""Compare the population of New York and Houston.  What is the percentage difference between the two populations?""")
        query = input('User: ')
        result = agent.run(query)
        print(f'Agent: {result}')

