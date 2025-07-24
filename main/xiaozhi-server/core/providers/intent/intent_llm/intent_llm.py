from typing import List, Dict
from ..base import IntentProviderBase
from plugins_func.functions.play_music import initialize_music_handler
from config.logger import setup_logging
import re, json, hashlib, time

TAG = __name__
logger = setup_logging()


class IntentProvider(IntentProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.llm = None
        self.promot = ""
        from core.utils.cache.manager import cache_manager, CacheType
        self.cache_manager = cache_manager
        self.CacheType = CacheType
        self.history_count = 4

    def get_intent_system_prompt(self, functions_list: List[Dict]) -> str:
        functions_desc = "Available functions list:\n"
        for func in functions_list:
            f = func.get("function", {})
            functions_desc += (
                f"\nFunction name: {f.get('name', '')}\n"
                f"Description: {f.get('description', '')}\n"
            )
            params = f.get("parameters", {}).get("properties", {})
            if params:
                functions_desc += "Parameters:\n"
                functions_desc += "\n".join(
                    f"- {k} ({v.get('type', '')}): {v.get('description', '')}"
                    for k, v in params.items()
                )
                functions_desc += "\n"
            functions_desc += "---\n"

        # Support for hass-mcp and mcp-proxy included in prompt instructions
        return(
            "You are an intent recognition assistant. Please analyze the user's last sentence, determine the user's intent, and call the corresponding function.\n\n"
            "- If the user uses question words (such as 'how', 'why', 'how') to ask questions related to exiting (for example, 'how did it exit?'), note that this is not asking you to exit, please return {'function_call': {'name': 'continue_chat'}}\n"
            "- Only trigger handle_exit_intent when the user explicitly uses commands such as 'exit system', 'end conversation', 'I don't want to talk to you anymore'\n\n"
            f"{functions_desc}\n"
            "If the user's request is related to controlling smart home devices (such as lights, switches, sensors, scenes, automations, media players, etc.), select the most appropriate function from the list (including hass-mcp and mcp-proxy functions) and generate the correct function_call. If the request mentions multiple actions (like “turn on the living room light and play music”), return an array of function_calls, each with the right name and parameters. Always use the function name and required arguments as described in the available functions.\n"
            "Processing steps:\n"
            "1. Analyze user input to determine user intent\n"
            "2. Select the most matching function from the list of available functions\n"
            "3. If a matching function is found, generate the corresponding function_call format\n"
            "4. If no matching function is found, return {\"function_call\": {\"name\": \"continue_chat\"}}\n\n"
            "Return format requirements:\n"
            "1. Must return in pure JSON format\n"
            "2. Must contain the function_call field\n"
            "3. function_call must contain the name field\n"
            "4. If the function requires parameters, it must contain the arguments field\n\n"
            "Example:\n"
            "```\n"
            "User: What time is it now?\n"
            "Return: {\"function_call\": {\"name\": \"get_time\"}}\n"
            "```\n"
            "```\n"
            "User: What is the current battery level?\n"
            "Return: {\"function_call\": {\"name\": \"get_battery_level\", \"arguments\": {\"response_success\": \"The current battery level is {value}%\", \"response_failure\": \"Unable to get the current battery level percentage of Battery\"}}}}\n"
            "```\n"
            "```\n"
            "User: What is the current screen brightness?\n"
            "Return: {\"function_call\": {\"name\": \"self_screen_get_brightness\"}}\n"
            "```\n"
            "```\n"
            "User: Set the screen brightness to 50%\n"
            "Return: {\"function_call\": {\"name\": \"self_screen_set_brightness\", \"arguments\": {\"brightness\": 50}}}}\n"
            "```\n"
            "```\n"
            "User: I want to end the conversation\n"
            "Return: {\"function_call\": {\"name\": \"handle_exit_intent\", \"arguments\": {\"say_goodbye\": \"goodbye\"}}}}\n"
            "```\n"
            "```\n"
            "User: Hello\n"
            "Return: {\"function_call\": {\"name\": \"continue_chat\"}}\n"
            "```\n\n"
            "Note:\n"
            "1. Only return in JSON format, do not include any other text\n"
            "2. If no matching function is found, return {\"function_call\": {\"name\": \"continue_chat\"}}\n"
            "3. Make sure the returned JSON format is correct and contains all necessary fields\n"
            "Special instructions:\n"
            "- When the user's single input contains multiple commands (such as 'turn on the light and turn up the volume')\n"
            "- Please return a JSON array composed of multiple function_calls\n"
            "- Example: {\'function_calls\': [{name:\'light_on\'}, {name:\'volume_up\'}]}\n"
        )

    def replyResult(self, text: str, original_text: str):
        return self.llm.response_no_stream(
            system_prompt=text,
            user_prompt="Please reply to the user in a human-like tone based on the above content, be concise, and return the result directly. The user is now saying:" + original_text,
        )

    async def detect_intent(self, conn, dialogue_history: List[Dict], text: str) -> str:
        if not self.llm:
            raise ValueError("LLM provider not set")
        if conn.func_handler is None:
            return '{"function_call": {"name": "continue_chat"}}'

        total_start = time.time()
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"Using model: {model_info}")

        cache_key = hashlib.md5((conn.device_id + text).encode()).hexdigest()
        if (cached := self.cache_manager.get(self.CacheType.INTENT, cache_key)):
            logger.bind(tag=TAG).debug(f"Cache hit: {cache_key}, cost: {time.time() - total_start:.4f}s")
            return cached

        # Generate system prompt if not already built
        if not self.promot:
            functions = conn.func_handler.get_functions() or []
            if (tools := getattr(conn, "mcp_client", None)):
                mcp_tools = tools.get_available_tools()
                if mcp_tools:
                    functions += mcp_tools
            # Add mcp-proxy support
            if (proxy_tools := getattr(conn, "mcp_proxy_client", None)):
                mcp_proxy_tools = proxy_tools.get_available_tools()
                if mcp_proxy_tools:
                    functions += mcp_proxy_tools
            self.promot = self.get_intent_system_prompt(functions)

        prompt = self.promot

        # Append music info
        music_cfg = initialize_music_handler(conn)
        prompt += f"\n<musicNames>{music_cfg.get('music_file_names', '')}</musicNames>"

        # Append Home Assistant device info
        hass_cfg = conn.config["plugins"].get("home_assistant", {})
        devices = hass_cfg.get("devices", [])
        if devices:
            prompt += "\nHere is a list of smart home devices:\n" + "\n".join(devices)

        logger.bind(tag=TAG).debug(f"Final prompt: {prompt}")

        # Build user dialogue history
        history_text = "\n".join(
            f"{msg.role}: {msg.content}" for msg in dialogue_history[-self.history_count:]
        )
        user_prompt = f"current dialogue:\n{history_text}\nUser: {text}"

        llm_start = time.time()
        intent_raw = self.llm.response_no_stream(system_prompt=prompt, user_prompt=user_prompt)
        logger.bind(tag=TAG).debug(f"LLM call time: {time.time() - llm_start:.4f}s")

        # Try to extract and clean JSON from response
        intent_raw = intent_raw.strip()
        match = re.search(r"\{.*\}", intent_raw, re.DOTALL)
        intent_json = match.group(0) if match else intent_raw

        try:
            parsed = json.loads(intent_json)
            self.cache_manager.set(self.CacheType.INTENT, cache_key, intent_json)

            if "function_call" in parsed:
                name = parsed["function_call"].get("name")
                args = parsed["function_call"].get("arguments", {})
                logger.bind(tag=TAG).info(f"LLM recognized intent: {name}, parameters: {args}")

                if name == "continue_chat":
                    conn.dialogue.dialogue = [
                        msg for msg in conn.dialogue.dialogue if msg.role not in ["tool", "function"]
                    ]
            return intent_json
        except json.JSONDecodeError:
            logger.bind(tag=TAG).error(f"Intent JSON parse error: {intent_raw}")
            return '{"function_call": {"name": "continue_chat"}}'
