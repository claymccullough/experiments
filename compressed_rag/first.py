# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from pprint import pprint
from typing import Dict, Any, Optional, List
from uuid import UUID
import os

import requests
from langchain.agents import initialize_agent, AgentType, create_structured_chat_agent
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_core.agents import AgentAction
from langchain_core.tools import Tool

from dotenv import load_dotenv

load_dotenv('.env')

from langchain.chains import create_retrieval_chain, RetrievalQA, LLMMathChain, ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler, BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

from models.models import get_ollama_model

"""
GLOBALS
"""
COMPRESS_BASE_URL = os.environ.get('COMPRESS_BASE_URL')


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        prompts = ['tell me a joke']
        # test = 'hi'

    def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        test = 'hi'

    def on_agent_action(
            self,
            action: AgentAction,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        test = 'hi'


load_dotenv('.env')

import os

from langchain_community.vectorstores.chroma import Chroma

INFER_MODEL_NAME = os.environ.get("INFER_MODEL_NAME")
INFER_BASE_URL = os.environ.get("INFER_BASE_URL")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL")


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


def compressed_query(llm, retriever, query: str):
    # step 1: get relevant docs
    docs = retriever.get_relevant_documents(query)
    print(len(docs))
    print(docs)
    question = f"Question: {query}\nHelpful Answer:"
    instruction = "You are a helpful assistant.  Use only information provided below without other sources.  If you don't know the answer, say \"I don't know\"."
    context = [doc.page_content for doc in docs]
    prompt = f"""
    {instruction}
    
    {context}
    
    {question}
    """

    # step 2: compress prompt
    result = requests.post(f'{COMPRESS_BASE_URL}/prompt_compressor', json={
        "context": context,
        "instruction": instruction,
        "question": question,
        "ratio": 0.5,
        # "target_token": 300
    })
    final = result.json()
    compressed_prompt = final['compressed_prompt']
    del final['compressed_prompt']
    pprint(final)

    # step 3: invoke LLM
    return llm.invoke(compressed_prompt)
    # return llm.invoke(prompt), llm.invoke(compressed_prompt)


if __name__ == '__main__':
    print('COMPRESSED RAG')
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL)
    vector = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    # Define a retriever interface
    retriever = vector.as_retriever()

    # Define LLM
    llm, stats_handler = get_ollama_model()

    # # Create a retrieval chain to answer questions
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     # return_source_documents=True,
                                     callbacks=[CustomHandler()],
                                     verbose=True)

    while True:
        # agent.run("""Compare the population of New York and Houston.  What is the percentage difference between the two populations?""")
        query = input('User: ')
        # result, compressed_result = compressed_query(llm=llm, retriever=retriever, query=query)
        result = compressed_query(llm=llm, retriever=retriever, query=query)
        print(f'Agent: {result}')
        # print(f'Compressed Agent: {compressed_result}')
        pprint(stats_handler.get_stats())
        stats_handler.reset()
