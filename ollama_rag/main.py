import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field


from dotenv import load_dotenv
from langchain.chains import RetrievalQA, LLMMathChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

load_dotenv('.env')

import os

from langchain_community.vectorstores.chroma import Chroma

MODEL_NAME = os.environ.get("MODEL_NAME")



primes = {998: 7901, 999: 7907, 1000: 7919}


class CalculatorInput(BaseModel):
    question: str = Field()


class PrimeInput(BaseModel):
    n: int = Field()


def is_prime(n: int) -> bool:
    if n <= 1 or (n % 2 == 0 and n > 2):
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_prime(n: int, primes: dict = primes) -> str:
    return str(primes.get(int(n)))


async def aget_prime(n: int, primes: dict = primes) -> str:
    return str(primes.get(int(n)))


if __name__ == '__main__':
    print('RAG')
    # load from disk
    embedding_function = OllamaEmbeddings(model=MODEL_NAME, base_uri=)
    vector = Chroma(persist_directory="./cities_chroma_db", embedding_function=embedding_function)

    # Define a retriever interface
    retriever = vector.as_retriever()

    # Define LLM
    llm = Ollama(model="mistral",
                 temperature=0,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name="GetPrime",
            func=get_prime,
            description="A tool that returns the `n`th prime number",
            args_schema=PrimeInput,
            coroutine=aget_prime,
        ),
        Tool.from_function(
            func=llm_math.run,
            name="Calculator",
            description="Useful for when you need to compute mathematical expressions",
            args_schema=CalculatorInput,
            coroutine=llm_math.arun,
        ),
    ]

    # Create agent
    agent = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    question = "What is the product of the 998th, 999th and 1000th prime numbers?"

    for step in agent_executor.iter({"input": question}):
        if output := step.get("intermediate_step"):
            action, value = output[0]
            if action.tool == "GetPrime":
                print(f"Checking whether {value} is prime...")
                assert is_prime(int(value))
            # Ask user if they want to continue
            _continue = input("Should the agent continue (Y/n)?:\n") or "Y"
            if _continue.lower() != "y":
                break
