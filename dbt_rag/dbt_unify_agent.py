from dotenv import load_dotenv

from semantic_routing.dbt_routing import get_route_layer

load_dotenv('.env')

import streamlit as st

import os
from typing import List

from langchain_community.callbacks import StreamlitCallbackHandler, get_openai_callback
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chains import LLMMathChain, RetrievalQA
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from retrievers.retriever import get_vector_retriever

OPENAI_MN = os.environ.get('OPENAI_MN')

dbt_tool_details = [
    {
        'name': 'metadata',
        'k': 10,
        'description': """
        This tool contains metadata about a dbt model. This metadata is
        helpful for answering questions about the type of model, which schema and owner
        it belong to, as well as how many bytes and rows the model contains.  It also
        can answer when the model was created and what it's unique id is.
        It can answer questions like:
        - "What type of dbt model is this?",
        - "What schema does this dbt model belong to?",
        - "How many bytes does this dbt model contain?",
        - "How many rows does this dbt model have?",
        - "When was this dbt model created?",
        - "What is the unique id of this dbt model?"
        """
    },
    {
        'name': 'code',
        'k': 10,
        'description': """
        This Tool contains the raw SQL code used by the dbt model to generate the table.
        This is really helpful for understanding the implementation of the dbt model's table, as well as source tables
        that were joined to generate it.  It also contains information on which conditions in the SQL may be
        filtering certain rows present in the dbt model's table.  It can answer questions about which post_hooks the dbt model has as well.
        It can answer questions like:
        - "How were tables joined to generate this DBT model?"
        - "What post_hooks does this dbt model have?"
        - "Explain the logic behind the DBT model."
        - "How are rows filtered in the DBT model?"
        - "Is this DBT model an incremental model?"
        - "What source tables were used to generate this DBT model?"
        """
    },
    {
        'name': 'columns',
        'k': 4,
        'description': """
        This Tool contains information on the columns inside the dbt model, including all column names,
        column types, and even descriptions around what each column is for. It also answers how many columns
        the dbt model has. It can answer questions like:
        - "How many columns does this dbt model have?"
        - "What is the description for this column on this dbt model?"
        - "What is the data type for this column on this dbt model?"
        - "Does this DBT model have a column like this?"
        - "Does this DBT model have any date columns?"
        - "What is the primary key of ths DBT model?"
        """
    },
    {
        'name': 'descriptions',
        'k': 4,
        'description': """
        This Tool contains a more detailed description of the dbt model, as well as some specific
        Spark and DBT related configuration for the table. It can answer questions like:
        - "What is the spark configuration for this dbt model?"
        - "What is the location of this dbt model?"
        - "What Serde library is used for this dbt model?"
        - "What is the input format for this dbt model?"
        - "What is the output format for this dbt model?"
        - "What are the statistics for this dbt model?"
        """
    },
    {
        'name': 'upstream_models',
        'k': 4,
        'description': """
        This Tool lists the models that are upstream to the dbt model, indicating that
        this model is a child model to all its upstream models.  This information can be used to figure out
        which upstream models may impact the dbt model if they were to be changed in any way. It can answer questions like:
        - "What upstream models does this dbt model have?"
        - "Does this dbt model have any upstream models?"
        - "Is this dbt model upstream to that dbt model?"
        - "How many upstream models does this dbt model have?"
        - "Check the upstream model flow for this dbt model and that dbt model. Will that dbt model be impacted if this dbt model changes?"
        - "Does this dbt model have any upstream models named like this?"
        """
    },
    {
        'name': 'downstream_models',
        'k': 4,
        'description': """
        This document lists the models that are downstream from the dbt model, indicating that
        this model is a source model to all its downstream models.  This information can be used to figure out
        which models may be affected if this dbt model were to be changed in any way.  It can answer questions like:
        - "What downstream models does this dbt model have?"
        - "Does this dbt model have any downstream models?"
        - "Is this dbt model downstream to that dbt model?"
        - "How many downstream models does this dbt model have?"
        - "Check the downstream model flow for this dbt model from that dbt model. Will this dbt model be impacted if that dbt model changes?"
        - "Does this dbt model have any downstream models named like this?"
        """
    },
]

def get_dbt_tools(llm) -> List[Tool]:
    # Create a retrieval chain to answer questions
    return [
        Tool(
            name=dbt_tool_detail.get('name'),
            func=RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=get_vector_retriever(
                    file_path=f'./embeddings/dbt_{dbt_tool_detail["name"]}',
                    k=dbt_tool_detail['k']
                ),
                verbose=True
            ).run,
            description=dbt_tool_detail.get('description')
        )
        for dbt_tool_detail in dbt_tool_details
    ]


if __name__ == '__main__':
    print('UNIFIED AGENT')
    llm = ChatOpenAI(
        verbose=True,
        temperature=0.0,
        model_name=OPENAI_MN,
    )
    llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        *get_dbt_tools(llm=llm),
        Tool(
            name="llm_math",
            func=llm_math.run,
            description="Useful for when you need to answer math questions."
        )
    ]

    system_message = f"""
    # CONTEXT #
    You are an expert data engineer with thorough knowledge of the DBT project at hand.  You have several tools at your
    disposal to find the information you need about the dbt project, and you respond to any question you are asked about it.
    If one of your tools fails to answer your question and you do not know the answer, call the remaining tools in this order: 
    metadata, code, descriptions, columns. If none of the tools can answer or the answers do not make sense. 
    then return "I don't know".
    
    #############
    
    # OBJECTIVE #
    Think step-by-step.  Analyze the context you are given to answer questions to the best of your acquired knowledge.
    
    #############
    
    # STYLE #
    Write using simple language, explaining your answers as simply as you can.
    
    #############
    
    # TONE #
    Maintain the same tone as the text supplied.
    
    #############
    
    # AUDIENCE #
    Your audience is generally curious about the resources in the DBT project and will ask questions around the models, 
    sources and columns in the DBT project.
    
    #############
    
    # RESPONSE #
    Finally, keep the response concise and succinct. Do not say "Based on the context provided", or "Based on the context given", just answer the question.
    If you don't know the answer or it does not make sense from the provided context, don't make up an answer.  Simply say 
    'I do not know.' Double-check that the model referenced in the answer matches the question's model EXACTLY.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message
            ),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5,
        early_stopping_method='generate',
    )

    route_layer = get_route_layer()
    if question := st.chat_input():
        st.chat_message('user').write(question)
        with st.chat_message('assistant'):
            st_callback = StreamlitCallbackHandler(st.container())

            # OpenAI
            with get_openai_callback() as cb:
                route = route_layer(question).name
                final_question = question
                st.write(f'Routing to: {route}')
                if route is not None:
                    final_question = f'{question} (SYSTEM NOTE: {route} tool will be helpful answering this question.)'
                response = agent_executor.invoke({"input": final_question}, {"callbacks": [st_callback]})
                st.write(response['output'])
                st.write(cb)
