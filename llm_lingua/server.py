from fastapi import FastAPI
from llmlingua import PromptCompressor
from pydantic import BaseModel

"""
GLOBALS
"""
app = FastAPI()
llm_lingua = PromptCompressor(model_name="openai-community/gpt2", device_map="cuda")


class PromptCompressionRequest(BaseModel):
    context: list[str]
    instruction: str
    question: str
    ratio: float
    target_token: float


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/prompt_compressor")
async def prompt_compressor(prompt_body: PromptCompressionRequest) -> dict:
    compressed_prompt = llm_lingua.compress_prompt(
        prompt_body.context,
        prompt_body.instruction,
        prompt_body.question,
        prompt_body.ratio,
        prompt_body.target_token
    )
    return compressed_prompt
