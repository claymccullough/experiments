# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import threading
from typing import Any, Dict, List, Optional
from uuid import UUID

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import initialize_agent, AgentType, create_structured_chat_agent, AgentExecutor, \
    create_openai_tools_agent, create_self_ask_with_search_agent
from langchain.chains import RetrievalQA
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks import StreamlitCallbackHandler, OpenAICallbackHandler, get_openai_callback
from langchain_community.callbacks.openai_info import standardize_model_name, MODEL_COST_PER_1K_TOKENS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.agents import AgentAction
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler, BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import Tool

load_dotenv('.env')

import os

from langchain_community.vectorstores.chroma import Chroma

DEST_PATH = os.environ.get('DEST_PATH')
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")
EMBED_BASE_URL = os.environ.get('EMBED_BASE_URL')
INFER_MODEL_NAME = os.environ.get("INFER_MODEL_NAME")
INFER_BASE_URL = os.environ.get('INFER_BASE_URL')

OpenAICallbackHandler()


class OllamaStatsHandler(BaseCallbackHandler):
    total_duration = 0
    start_input_tokens = 0
    total_input_tokens = 0
    final_output_tokens = 0
    total_output_tokens = 0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    """
    : get this completely working for OLLAMA
    """

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        if response.generations is None or len(response.generations) < 1 or len(response.generations[0]) < 1:
            return None

        generation_info = response.generations[0][0].generation_info
        input_tokens = generation_info.get('prompt_eval_count', 0)
        output_tokens = generation_info.get('eval_count', 0)
        total_duration = generation_info.get('total_duration', 0)

        # update shared state behind lock
        with self._lock:
            if self.start_input_tokens <= 0:
                self.start_input_tokens = input_tokens

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.final_output_tokens = output_tokens  # always override so we have the last one.
            self.total_duration += total_duration

    def get_stats(self):
        return {
            'total_duration': f'{self.total_duration / 1000000000:.4f} seconds',
            'input_tokens_start': self.start_input_tokens,
            'input_tokens_total': self.total_input_tokens,
            'output_tokens_end': self.final_output_tokens,
            'output_tokens_total': self.total_output_tokens,
        }


def get_ollama_model(stats_handler=OllamaStatsHandler()):
    return Ollama(
        model=INFER_MODEL_NAME,
        base_url=INFER_BASE_URL,
        temperature=0.0,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler(), stats_handler])
    )


def get_vector_retriever():
    # Define embedding function
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL)
    vector = Chroma(persist_directory=DEST_PATH, embedding_function=embedding_function)

    # Define a retriever interface
    return vector.as_retriever()


def get_qa_chain(llm):
    # Define a retriever interface
    retriever = get_vector_retriever()

    # Create a retrieval chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           # return_source_documents=True,
                                           verbose=True)
    return qa_chain



def get_rag_agent():
    # Define LLM
    stats_handler = OllamaStatsHandler()
    llm = get_ollama_model(stats_handler=stats_handler)

    # Create a retrieval chain to answer questions
    qa_chain = get_qa_chain(llm=llm)

    # Define tools
    tools = [
        Tool(
            name="general-knowledge",
            func=qa_chain.run,
            description="useful for when you need to answer questions about the documents in the database"
        ),
    ]

    # Define prompt template
    template = """
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "input"]
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=prompt,
        handle_parsing_errors=True,
        verbose=True,
    )
    return agent, stats_handler


if __name__ == '__main__':
    print('RAG')

    # llm = get_ollama_model()
    # retriever = get_vector_retriever()
    # result = retriever.get_relevant_documents('What is the population of New York City?')
    # context = ''.join([doc.page_content for doc in result])
    # print(context)


    agent, stats_handler = get_rag_agent()
    response = agent.invoke(
        {"input": "what is the population of New York City?"}
    )
    print(response)
    print(stats_handler.get_stats())
    #
    # agent, stats_handler = get_rag_agent()
    # response = agent.invoke(
    #     {"input": "what is the population of Houston TX"}
    # )
    # print(response)
    # print(stats_handler.get_stats())
    #
    # agent, stats_handler = get_rag_agent()
    # response = agent.invoke(
    #     {"input": "what is the population of New York City?"}
    # )
    # print(response)
    # print(stats_handler.get_stats())

    # if prompt := st.chat_input():
    #     agent, stats_handler = get_rag_agent()
    #     st.chat_message('user').write(prompt)
    #     with st.chat_message('assistant'):
    #         st_callback = StreamlitCallbackHandler(st.container())
    #         response = agent.invoke(
    #             {"input": prompt}, {"callbacks": [st_callback]}
    #         )
    #         st.write(response["output"])
    #         st.write(stats_handler.get_stats())
