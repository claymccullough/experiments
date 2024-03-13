import os

import pyperclip
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAI

from models.models import get_ollama_model

load_dotenv('.env')

import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler, get_openai_callback

from retrievers.retriever import get_vector_retriever, get_vector, compress_prompt

"""
GLOBALS
"""
INGEST_NAME = os.environ.get('INGEST_NAME')

retriever = get_vector_retriever(file_path=f'./embeddings/{INGEST_NAME}')
# meta_retriever = get_vector_retriever(file_path=f'./embeddings/dbt_metadata')
# col_retriever = get_vector_retriever(file_path=f'./embeddings/dbt_columns')
# retriever = EnsembleRetriever(retrievers=[meta_retriever, col_retriever])
vector = get_vector(file_path=f'./embeddings/{INGEST_NAME}')


def gen_prompt(question: str):
    docs = retriever.get_relevant_documents(question)
    page_content = [doc.page_content for doc in docs]
    context = "\n\n".join(page_content)
    prompt =  f"""
    # CONTEXT #
    {context}
    
    #############
    
    # OBJECTIVE #
    Think step-by-step.  Analyze the given context above to answer the question ```{question}``` between three backticks 
    to the best of your acquired knowledge.
    
    #############
    
    # STYLE #
    Write using simple language, explaining your answers as simply as you can.
    
    #############
    
    # TONE #
    Maintain the same tone as the text supplied.
    
    #############
    
    # AUDIENCE #
    Our audience is generally curious about the resources in the DBT project and will ask questions around the models, 
    sources and columns in the DBT project.
    
    #############
    
    # RESPONSE #
    Finally, keep the response concise and succinct. Do not say "Based on the context provided", or "Based on the context given", just answer the question.
    If you don't know the answer or cannot deduce it from the supplied Context, don't make up an answer.  Simply say 
    'I do not know.' Double-check that the model referenced in the answer matches the question's model EXACTLY.
    
    Answer:"""
    return {
        'prompt': prompt,
        'context': page_content,
    }


if __name__ == '__main__':
    print('OpenAI DBT RAG')
    # llm = OpenAI()
    llm, stats_handler = get_ollama_model()

    # question = "Tell me the schema for admin_backfill_days model"
    # question = "When was the model.limeade_lakehouse.sil_lo_activity_templates model created?"
    # question = "When was source.limeade_lakehouse.bronze_tenantstore.tenants_collection created?"
    # question = "How many rows does limeadedotcom_events have?"
    # docs = retriever.get_relevant_documents(question)
    # context = "\n\n".join([doc.page_content for doc in docs])
    # print(f'QUESTION: {question}')
    # print(context)

    # prompt = gen_prompt('When was the `model.limeade_lakehouse.sil_lo_activity_templates` model created?')
    # prompt = gen_prompt('What upstream models does the `model.limeade_lakehouse.sil_ai_mobile_events_slim` model have?')
    # pyperclip.copy(prompt)
    # gen_prompt('tell me more about the model.limeade_lakehouse.sil_ls_programs model')
    # gen_prompt('sil_ls_programs model')
    # gen_prompt('model.limeade_lakehouse.sil_lp_limeade_users')
    # gen_prompt('sil_lp_limeade_users')
    # gen_prompt('How many bytes does the `model.limeade_lakehouse.sil_ls_programs` model consume?')

    if question := st.chat_input():
        st.chat_message('user').write(question)
        with st.chat_message('assistant'):
            st_callback = StreamlitCallbackHandler(st.container())
            prompt = gen_prompt(question)
            docs = retriever.get_relevant_documents(question)
            context = "\n\n\n".join([doc.page_content for doc in docs])
            # pyperclip.copy(prompt)
            # st.write('COPIED')
            response = llm.invoke(input=prompt, config={"callbacks": [st_callback]})
            st.write(response)
            st.write(f'######## CONTEXT ########\n\n{context}')
            # with get_openai_callback() as cb:
            #     response = llm.invoke(input=prompt, config={"callbacks": [st_callback]})
            #     st.write(cb)
            st.write(stats_handler.get_stats())
            stats_handler.reset()
