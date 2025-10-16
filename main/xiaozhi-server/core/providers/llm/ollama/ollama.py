from config.logger import setup_logging
from openai import OpenAI
from ollama import Client as OllamaClient
import json
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()

class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.api_key = config.get("api_key", None)

        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

        if self.api_key:
            # Use Ollama's Python client
            self.client = OllamaClient(
                host="https://ollama.com",  # OllamaClient doesn't use /v1
                headers={'Authorization': self.api_key}
            )
            self.use_ollama_client = True
        else:
            # Fallback to OpenAI-compatible client
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",
            )
            self.use_ollama_client = False

        self.is_qwen3 = self.model_name and self.model_name.lower().startswith("qwen3")

    def response(self, session_id, dialogue, **kwargs):
        try:
            if self.is_qwen3:
                dialogue = self._inject_no_think(dialogue)

            if self.use_ollama_client:
                for part in self.client.chat(self.model_name, messages=dialogue, stream=True):
                    yield part['message']['content']
                return

            responses = self.client.chat.completions.create(
                model=self.model_name, messages=dialogue, stream=True
            )
            yield from self._stream_chunks(responses)

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in response: {e}")
            yield "【Ollama服务响应异常】"

    def response_with_functions(self, session_id, dialogue, functions=None):
        try:
            if self.is_qwen3:
                dialogue = self._inject_no_think(dialogue)

            if self.use_ollama_client:
                # NOTE: Ollama's Client does not support function calling yet
                logger.bind(tag=TAG).warning("Function calling not supported in Ollama client")
                yield "【Function calling not supported】", None
                return

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=True,
                tools=functions,
            )
            yield from self._stream_chunks(stream, include_tools=True)

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in response_with_functions: {e}")
            yield f"【Ollama服务响应异常: {str(e)}】", None

    def _inject_no_think(self, dialogue):
        dialogue_copy = dialogue.copy()
        for i in range(len(dialogue_copy) - 1, -1, -1):
            if dialogue_copy[i]["role"] == "user":
                dialogue_copy[i]["content"] = "/no_think " + dialogue_copy[i]["content"]
                logger.bind(tag=TAG).debug("为qwen3模型添加/no_think指令")
                break
        return dialogue_copy

    def _stream_chunks(self, responses, include_tools=False):
        is_active = True
        buffer = ""

        for chunk in responses:
            try:
                delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                content = getattr(delta, "content", None)
                tool_calls = getattr(delta, "tool_calls", None)

                if include_tools and tool_calls:
                    yield None, tool_calls
                    continue

                if content:
                    buffer += content

                    while "<think>" in buffer and "</think>" in buffer:
                        pre = buffer.split("<think>", 1)[0]
                        post = buffer.split("</think>", 1)[1]
                        buffer = pre + post

                    if "<think>" in buffer:
                        is_active = False
                        buffer = buffer.split("<think>", 1)[0]

                    if "</think>" in buffer:
                        is_active = True
                        buffer = buffer.split("</think>", 1)[1]

                    if is_active and buffer:
                        if include_tools:
                            yield buffer, None
                        else:
                            yield buffer
                        buffer = ""

            except Exception as e:
                logger.bind(tag=TAG).error(f"Error processing chunk: {e}")
                if include_tools:
                    yield None, None
