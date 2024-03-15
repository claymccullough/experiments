from dotenv import load_dotenv
from langchain import hub

from models.models import get_ollama_model, get_chat_ollama_model
from semantic_routing.dbt_routing import get_route_layer

load_dotenv('.env')

import streamlit as st

import os
from typing import List

from langchain_community.callbacks import StreamlitCallbackHandler, get_openai_callback
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, initialize_agent, AgentType, \
    create_openai_tools_agent, create_structured_chat_agent, create_react_agent
from langchain.chains import LLMMathChain, RetrievalQA
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from retrievers.retriever import get_vector_retriever

OPENAI_MN = os.environ.get('OPENAI_MN')

dbt_tool_details = [
    {
        'name': 'dbt-metadata',
        'description': """This tool contains metadata about a dbt model. This metadata is
helpful for answering questions about the type of model, which schema and owner
it belong to, as well as how many bytes and rows the model contains.  It also
can answer when the model was created and what it's unique id is."""
    },
#     {
#         'name': 'dbt code',
#         'description': """This Tool contains the raw SQL code used by the dbt model to generate the table.
# This is really helpful for understanding the implementation of the dbt model's table, as well as source tables
# that were joined to generate it.  It also contains information on which conditions in the SQL may be
# filtering certain rows present in the dbt model's table.  It can answer questions about which post_hooks the dbt model has as well."""
#     },
#     {
#         'name': 'dbt columns',
#         'description': """This Tool contains information on the columns inside the dbt model, including all column names,
# column types, and even descriptions around what each column is for. It also answers how many columns the dbt model has."""
#     },
#     {
#         'name': 'dbt descriptions',
#         'description': """This Tool contains a more detailed description of the dbt model, as well as some specific
# Spark and DBT related configuration for the table."""
#     },
#     {
#         'name': 'dbt upstream models',
#         'description': """This Tool lists the models that are upstream to the dbt model, indicating that
# this model is a child model to all its upstream models.  This information can be used to figure out
# which upstream models may impact the dbt model if they were to be changed in any way."""
#     },
#     {
#         'name': 'dbt downstream models',
#         'description': """This document lists the models that are downstream from the dbt model, indicating that
# this model is a source model to all its downstream models.  This information can be used to figure out
# which models may be affected if this dbt model were to be changed in any way."""
#     },
]


def get_dbt_tools(llm) -> List[Tool]:
    # Create a retrieval chain to answer questions
    return [
        Tool(
            name=dbt_tool_detail.get('name'),
            func=RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=get_vector_retriever(file_path=f'./embeddings/{dbt_tool_detail["name"]}'),
                verbose=True
            ).run,
            description=dbt_tool_detail.get('description')
        )
        for dbt_tool_detail in dbt_tool_details
    ]


if __name__ == '__main__':
    print('UNIFIED AGENT')

    llm, stats_handler = get_chat_ollama_model()
    llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        *get_dbt_tools(llm=llm),
        Tool(
            name="llm math",
            func=llm_math.run,
            description="Useful for when you need to answer math questions."
        )
    ]

    # Define prompt template
    system_message = """
    # CONTEXT #
    You are an expert data engineer with thorough knowledge of the DBT project at hand.  You have several tools at your
    disposal to find the information you need about the dbt project, and you respond to any question you are asked about it.
    
    These are the tools you have access to: {tool_names}
    
    Here is more detail on the tools:
    {tools}
    
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
    If you don't know the answer or cannot deduce it from the supplied context, don't make up an answer.  Simply say 
    'I do not know.' Double-check that the model referenced in the answer matches the question's model EXACTLY.
    """
    # prompt = hub.pull("hwchase17/react")
    prompt = ChatPromptTemplate.from_template(
        template="""
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
    )
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             system_message
    #         ),
    #         # MessagesPlaceholder(variable_name="chat_history"),
    #         ("human", "{input}"),
    #         MessagesPlaceholder(variable_name="agent_scratchpad"),
    #         MessagesPlaceholder(variable_name="intermediate_steps"),
    #     ]
    # )
    agent = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        # handle_parsing_errors="Check you output and make sure it conforms! Do not output an action and a final answer at the same time."
        handle_parsing_errors=True
    )

    # agent = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors=True,
    #     # prompt=prompt
    # )
    # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     verbose=True,
    #     return_intermediate_steps=True
    # )

    route_layer = get_route_layer()
    question = 'How many rows does the `model.limeade_lakehouse.sil_ai_identity_events` model have?'
    # question = 'What is 2+2?'
    route = route_layer(question).name
    print(f'ROUTE: {route}')
    result = agent_executor.invoke({'input': question})
    print(result)
    # print(result)
    # test = 'docs'

    # # result = route_layer('What schema does this `model.limeade_lakehouse.sil_ai_identity_events` model belong to?')
    # # print(result)
    # #
    # if question := st.chat_input():
    #     st.chat_message('user').write(question)
    #     with st.chat_message('assistant'):
    #         st_callback = StreamlitCallbackHandler(st.container())
    #
    #         # Local Mistral Instruct
    #         route = route_layer(question).name
    #         final_question = question
    #         if route is not None:
    #             st.write(f'Routing to: {route}')
    #             # final_question = f'Use the {route} tool to answer the following question: {question}'
    #         response = agent.invoke({"input": final_question}, {"callbacks": [st_callback]})
    #         st.write(response['output'])
    #         st.write(stats_handler.get_stats())
    #         stats_handler.reset()
    #
    #         # OpenAI
    #         # with get_openai_callback() as cb:
    #         #     route = route_layer(question).name
    #         #     final_question = question
    #         #     if route is not None:
    #         #         st.write(f'Routing to: {route}')
    #         #         final_question = f'According to {route}, {question}'
    #         #     response = agent_executor.invoke({"input": final_question}, {"callbacks": [st_callback]})
    #         #     st.write(response['output'])
    #         #     st.write(cb)
