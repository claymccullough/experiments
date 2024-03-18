import json
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langgraph.graph import StateGraph, END

from semantic_routing.dbt_routing import get_route_layer

load_dotenv('.env')

import streamlit as st

import os
from typing import List, TypedDict, Annotated, Sequence

from langchain_community.callbacks import StreamlitCallbackHandler, get_openai_callback
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chains import LLMMathChain, RetrievalQA
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from retrievers.retriever import get_vector_retriever

OPENAI_MN = os.environ.get('OPENAI_MN')
route_layer = get_route_layer()


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


@tool
def query_routing_tool(query: str):
    """Given the human query, return the name of the tool to search"""
    return route_layer(query).name


def supervisor(state):
    human_input = state["messages"][-1].content
    route_name = route_layer(human_input).name
    if route_name:
        ai_message = f"I am going to use the {route_name} tool to answer this question: {human_input}"
    else:
        ai_message = f"The query_routing tool was not able to route the query.  I will try to use the tool that makes the most sense for this question: {human_input}"
    return {
        "messages": [
            AIMessage(
                content=ai_message,
            )
        ]
    }


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
        streaming=True
    )
    llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        *get_dbt_tools(llm=llm),
        query_routing_tool,
        Tool(
            name="llm_math",
            func=llm_math.run,
            description="Useful for when you need to answer math questions."
        )
    ]
    tool_executor = ToolExecutor(tools=tools)
    functions = [format_tool_to_openai_function(t) for t in tools]
    llm = llm.bind_functions(functions)

    """
    Flow:
    1. On first run, Supervisor gets routed answer.  Subsequent runs, chooses best tool remaining for question at hand until there are no tools left.
    2. Agent takes tool and query from supervisor and answers question.
    3. Supervisor reviews response, makes sure question is answered.  If not, return to Step 1.
    """

    """
    https://python.langchain.com/docs/langgraph
    """
    class AgentState(dict):
        messages: Annotated[Sequence[BaseMessage], operator.add]


    # Define the function that determines whether to continue or not
    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            return "end"

        # Otherwise if there is, we continue
        else:
            return "continue"

    # Define the function that calls the model
    def call_model(state):
        messages = state['messages']
        response = llm.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the function to execute tools
    def call_tool(state):
        messages = state['messages']
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
        )
        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}


    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("route_first", supervisor)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("route_first")
    workflow.add_edge('route_first', 'agent')

    # We now add a conditional edge

    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END
        }
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge('action', 'agent')

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    # inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
    # inputs = {"messages": [HumanMessage(content="How many rows does the `model.limeade_lakehouse.sil_ai_identity_events` model have?")]}
    # question = "How long is the `model.limeade_lakehouse.sil_ai_identity_events` model set to retain when it VACUUMs?"
    # question = "How many columns does the `model.limeade_lakehouse.sil_upmc_recommendations` model have?"
    question = "Does the `model.limeade_lakehouse.sil_lp_user_activities` model have a post_hook that runs OPTIMIZE on the table?"
    inputs = {"messages": [
        SystemMessage(content=f"""
        You are an expert data engineer with thorough knowledge of the DBT project at hand.  You have several tools at your
        disposal to find the information you need about the dbt project, and you respond to any question you are asked about it.  
        If you don't know the answer or it does not make sense from the provided context, don't make up an answer.  Simply say
        'I do not know.' Double-check that the model referenced in the answer matches the question's model EXACTLY.
        """),
        HumanMessage(content=question)
    ]}
    for output in app.stream(inputs):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")


    # route_layer = get_route_layer()
    # if question := st.chat_input():
    #     st.chat_message('user').write(question)
    #     with st.chat_message('assistant'):
    #         st_callback = StreamlitCallbackHandler(st.container())
    #
    #         # OpenAI
    #         with get_openai_callback() as cb:
    #             route = route_layer(question).name
    #             final_question = question
    #             st.write(f'Routing to: {route}')
    #             if route is not None:
    #                 final_question = f'{question} (SYSTEM NOTE: {route} tool will be helpful answering this question.)'
    #             response = agent_executor.invoke({"input": final_question}, {"callbacks": [st_callback]})
    #             st.write(response['output'])
    #             st.write(cb)
