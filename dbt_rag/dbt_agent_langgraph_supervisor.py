import operator
import operator
import random

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langgraph.graph import StateGraph, END

from semantic_routing.dbt_routing import get_route_layer
from utilities.evals import eval_response

load_dotenv('.env')

import streamlit as st

import os
from typing import List, Annotated, Sequence

from langchain_community.callbacks import StreamlitCallbackHandler, get_openai_callback
from langchain.chains import LLMMathChain, RetrievalQA
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from retrievers.retriever import get_vector_retriever

OPENAI_MN = os.environ.get('OPENAI_MN')
EVAL_OPENAI_MN = os.environ.get('EVAL_OPENAI_MN')
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
dbt_tool_detail_lookup = {
    dbt_tool['name']: dbt_tool['description']
    for dbt_tool in dbt_tool_details
}


@tool
def query_routing_tool(query: str):
    """Given the human query, return the name of the tool to search"""
    return route_layer(query).name


class SupervisorState(dict):
    query: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str
    used_tools: List[str]
    remaining_tools: List[str]


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


def get_next_action(llm, question: str, remaining_tools: list):
    if len(remaining_tools) < 1:
        return None, []

    # TODO: use LLM to get next selected tool
    tools = [
        {
            'name': name,
            'description': dbt_tool_detail_lookup[name]
        }
        for name in remaining_tools
    ]
    prompt = f"""
        Given the tool definitions here:
        
        {tools}
        
        I want to answer this question:
        {question}
        
        Select the tool I should use and return just the name of the tool, don't explain why you chose the tool.
        
        Answer:"""
    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    # Ask the LLM which tool we should use next. If bad answer, select one at random.
    selected_tool = response.content.lower() \
        if (response and response.content and response.content.lower() in remaining_tools) \
        else random.choice(remaining_tools)
    return selected_tool, [
        remaining_tool
        for remaining_tool in remaining_tools
        if remaining_tool != selected_tool
    ]


def should_continue(state):
    # This function needs to do eval. this is the supervisor.
    return 'end' if state.get('next_action') == 'end' else 'continue'


if __name__ == '__main__':
    print('UNIFIED AGENT')
    eval_llm = ChatOpenAI(
        verbose=True,
        temperature=0.0,
        model_name=EVAL_OPENAI_MN,
    )
    utility_llm = ChatOpenAI(
        verbose=True,
        temperature=0.0,
        model_name=OPENAI_MN,
    )
    llm_math = LLMMathChain.from_llm(llm=utility_llm, verbose=True)
    tools = [
        *get_dbt_tools(llm=utility_llm),
    ]
    tool_executor = ToolExecutor(tools=tools)
    functions = [format_tool_to_openai_function(t) for t in tools]
    llm = utility_llm.bind_functions(functions)


    def supervisor(state: SupervisorState):
        # Get last message, eval to see if it answers the question. If so, end.
        last_message = state['messages'][-1]
        query = state.get('query') if state.get('query') else last_message.content
        used_tools = state.get('used_tools') if state.get('used_tools') else []
        remaining_tools = state['remaining_tools'] \
            if state['remaining_tools'] \
            else [dbt_tool['name'] for dbt_tool in dbt_tool_details]
        if last_message and last_message.type == "ai" and eval_response(llm=eval_llm, question=query,
                                                                        answer=last_message.content):
            return {'next_action': 'end'}

        # Given the remaining tools, use LLM to determine which tool should be used next.
        next_action, remaining_tools = get_next_action(llm=utility_llm, question=query, remaining_tools=remaining_tools)
        print(f'SELECTED TOOL: {next_action}')
        print(f'REMAINING TOOLS: {remaining_tools}')
        used_tools.append(next_action)
        if not next_action:
            return {'next_action': 'end'}

        # Return next state to agent.
        return {
            'query': query,
            'next_action': next_action,
            'used_tools': used_tools,
            'remaining_tools': remaining_tools
        }


    def agent(state):
        action = ToolInvocation(
            tool=state['next_action'],
            tool_input=state['query'],
        )
        response = tool_executor.invoke(action)
        return {"messages": [AIMessage(content=response)]}


    # Flow:
    # 1. On first run, Supervisor gets routed answer.  Subsequent runs, chooses best tool remaining for question at hand until there are no tools left.
    # 2. Agent takes tool and query from supervisor and answers question.
    # 3. Supervisor reviews response, makes sure question is answered.  If not, return to Step 1.
    # https://python.langchain.com/docs/langgraph
    workflow = StateGraph(SupervisorState)

    # Define the two nodes we will cycle between
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("agent", agent)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "continue": "agent",
            "end": END
        }
    )

    # Route back from agent to supervisor
    workflow.add_edge('agent', 'supervisor')

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()
    if question := st.chat_input():
        st.chat_message('user').write(question)
        with st.chat_message('assistant'):
            st_callback = StreamlitCallbackHandler(st.container())

            with get_openai_callback() as cb:
                response = app.invoke({
                    "messages": [HumanMessage(content=question)]
                }, {
                    "callbacks": [st_callback]
                })
                ai_answer = response['messages'][-1].content
                st.write(ai_answer)
                st.write('-' * 80)
                st.write(f'USED TOOLS: {response["used_tools"]}')
                st.write(f'REMAINING TOOLS: {response["remaining_tools"]}')
                st.write(cb)
