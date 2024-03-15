from uuid import UUID

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

load_dotenv('.env')

import os
import threading
from typing import Any, Dict, Optional, List

from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.outputs import LLMResult



"""
GLOBALS
"""
INFER_MODEL_NAME = os.environ.get("INFER_MODEL_NAME")
INFER_BASE_URL = os.environ.get("INFER_BASE_URL")


class OllamaStatsHandler(BaseCallbackHandler):
    total_duration = 0
    start_input_tokens = 0
    total_input_tokens = 0
    final_output_tokens = 0
    total_output_tokens = 0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.total_duration = 0
            self.start_input_tokens = 0
            self.total_input_tokens = 0
            self.final_output_tokens = 0
            self.total_output_tokens = 0

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
        # stop=[],
        callback_manager=CallbackManager([stats_handler])
    ), stats_handler


def get_chat_ollama_model(stats_handler=OllamaStatsHandler()):
    return ChatOllama(
        model=INFER_MODEL_NAME,
        base_url=INFER_BASE_URL,
        temperature=0.0,
        callback_manager=CallbackManager([stats_handler]),
    ), stats_handler