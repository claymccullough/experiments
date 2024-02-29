# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_react_agent
from langchain.chains import RetrievalQA, LLMMathChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter, DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.callbacks import StreamlitCallbackHandler, get_openai_callback, OpenAICallbackHandler
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.openai import OpenAI
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import Tool

load_dotenv('.env')

import os

from langchain_community.vectorstores.chroma import Chroma

DEST_PATH = os.environ.get('DEST_PATH')
MODEL_NAME = os.environ.get("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
BASE_URL = os.environ.get('BASE_URL')

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    print('Content Length:')
    print(len(''.join([source.page_content for source in llm_response["source_documents"]])))

# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

if __name__ == '__main__':
    """
    https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression
    """
    print('RAG')

    # Define LLM
    llm = OpenAI(
        model_name=MODEL_NAME,
        temperature=0,
        # streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler(), OpenAICallbackHandler()])
    )
    llm_math = LLMMathChain.from_llm(llm, verbose=True)
    compressor = LLMChainExtractor.from_llm(llm)

    # Define embedding function
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=BASE_URL)
    vector = Chroma(persist_directory=DEST_PATH, embedding_function=embeddings)
    retriever = vector.as_retriever()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    with get_openai_callback() as cb:
        query = "Who is the owner of the `model.jaffle_shop.dim_customers.v2` model?"
        docs = compression_retriever.get_relevant_documents(query)
        print(docs)
        print(cb)




    # splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    # redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    # relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    # pipeline_compressor = DocumentCompressorPipeline(
    #     transformers=[splitter, redundant_filter, relevant_filter]
    # )
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=pipeline_compressor, base_retriever=retriever
    # )
    # compressed_docs = compression_retriever.get_relevant_documents(
    #     "What did the president say about Ketanji Jackson Brown"
    # )
    # pretty_print_docs(compressed_docs)

    # Define prompt template
    # prompt = hub.pull("hwchase17/react")
    # template = """
    # Answer the following question based only on the provided context:
    #
    # <context>
    # {context}
    # </context>
    #
    # Question: {question}
    # """

    # Create a retrieval chain to answer questions
    # qa_chain = RetrievalQA.from_chain_type(llm=llm,
    #                                        chain_type="stuff",
    #                                        retriever=compression_retriever,
    #                                        # retriever=retriever,
    #                                        # return_source_documents=True,
    #                                        chain_type_kwargs={
    #                                            "prompt": PromptTemplate(
    #                                                template=template,
    #                                                input_variables=["context", "question"]
    #                                            )
    #                                        },
    #                                        verbose=True)
    #
    # tools = [
    #     Tool(
    #         name="dbt project",
    #         func=qa_chain.run,
    #         description="useful for when you need to answer questions about the dbt project"
    #     ),
    #     # Tool(
    #     #     name="llm-math",
    #     #     func=llm_math.run,
    #     #     description="Useful for when you need to answer questions about math."
    #     # )
    # ]
    #
    # agent = initialize_agent(
    #     tools=tools,
    #     llm=llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     max_iterations=5,
    #     verbose=True,
    #     return_intermediate_steps=True
    # )
    #
    # # Interact with RAG
    # with get_openai_callback() as cb:
    #     # query = """Who is the owner of the DIM_CUSTOMERS_V1 model?"""
    #     query = """When was the DIM_CUSTOMERS_V1 model created?"""
    #     # query = "How many bytes is the `model.jaffle_shop.fct_orders` table consuming?  Also give me the answer in KB"
    #     # query = "List all the models that have the FIRST_ORDER column. Also, what is the type of that column in each table respectively?" # BREAKS IT
    #     # query = "List out all the tables in the dbt project with 'raw' in the name"
    #     # query = "Sum the total bytes consumed by all the models in the dbt project. Also give me the answer in GB." # DOESNT DO WELL WITH AGGREGATION - may want to present it to the LLM up front.
    #     # query = "How many bytes are in the gold_fact_communications_messages_log table?"
    #     llm_response = agent(query)
    #     print(llm_response)
    #     print(cb)
