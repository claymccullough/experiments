import pyperclip
from dotenv import load_dotenv
from langchain_openai import OpenAI

from models.models import get_ollama_model

load_dotenv('.env')

import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler, get_openai_callback

from retrievers.retriever import get_vector_retriever, get_vector

retriever = get_vector_retriever(file_path='./embeddings/dbt-test')
vector = get_vector(file_path='./embeddings/dbt-test')


def gen_prompt(question: str):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"""
    # CONTEXT #
    Use the following relevant context from the DBT Project to answer the user's question:
    {context}
    
    #############
    
    # OBJECTIVE #
    Answer the question ```{question}``` between three backticks to the best of your acquired knowledge.  Think step-by-step. 
    If you don't know the answer or cannot deduce it from the supplied Context, don't make up an answer.  Simply say 'I do not know.'
    
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
    Finally, keep the response concise and succinct.
    """


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
            pyperclip.copy(prompt)
            st.write('COPIED')
            response = llm.invoke(input=prompt, config={"callbacks": [st_callback]})
            st.write(response)
            # with get_openai_callback() as cb:
            #     response = llm.invoke(input=prompt, config={"callbacks": [st_callback]})
            #     st.write(cb)
            st.write(stats_handler.get_stats())
            stats_handler.reset()
