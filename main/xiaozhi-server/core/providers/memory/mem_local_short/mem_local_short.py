from ..base import MemoryProviderBase, logger
import time
import json
import os
import yaml
from config.config_loader import get_project_dir
from config.manage_api_client import save_mem_local_short
from core.utils.util import check_model_key

short_term_memory_prompt = """
# Spatiotemporal Memory Weaver

## Core Mission
Build a dynamic, expandable memory network that retains key information within a limited space, while intelligently tracking the evolution of information.
Summarize important user information from the conversation history to enable more personalized service in future conversations.

## Memory Rules
### 1. Three-Dimensional Memory Evaluation (to be executed with each update)
| Dimension        | Evaluation Criteria                     | Weight |
|------------------|----------------------------------------|--------|
| Timeliness       | Freshness of information (by turn)      | 40%    |
| Emotional Intensity | Contains 💖 tag / Number of mentions | 35%    |
| Connection Density | Number of connections to other info   | 25%    |

### 2. Dynamic Update Mechanism
**Example of Name Change Handling:**
Original memory: `"Former Names": ["Zhang San"], "Current Name": "Zhang Sanfeng"`
Trigger condition: When a naming signal like "My name is X" or "Call me Y" is detected
Process:
1. Move the old name into the "Former Names" list
2. Record the name change on the timeline: "2024-02-15 14:32: Activated Zhang Sanfeng"
3. Add to the memory cube: "Identity transformation from Zhang San to Zhang Sanfeng"

### 3. Space Optimization Strategy
- **Information Compression:** Use symbolic systems to increase density  
  - ✅ "Zhang Sanfeng[Beijing/Software/🐱]"
  - ❌ "Beijing software engineer, cat owner"
- **Elimination Warning:** Trigger when total word count ≥ 900  
  1. Delete items with score < 60 not mentioned in 3 rounds
  2. Merge similar items (keep the latest timestamp)

## Memory Structure
The output format **must** be a parsable JSON string, with no explanations, comments, or examples. When saving memory, extract only information from the conversation—**do not mix in example content**.
```json
{
  "Spatiotemporal Archive": {
    "Identity Map": {
      "Current Name": "",
      "Feature Tags": []
    },
    "Memory Cube": [
      {
        "Event": "Joined a new company",
        "Timestamp": "2024-03-20",
        "Emotional Value": 0.9,
        "Related Items": ["Afternoon Tea"],
        "Shelf Life": 30
      }
    ]
  },
  "Relationship Network": {
    "Frequent Topics": {"Workplace": 12},
    "Hidden Connections": [""]
  },
  "Pending Responses": {
    "Urgent Matters": ["Tasks that require immediate action"],
    "Potential Care": ["Help that can be proactively provided"]
  },
  "Highlight Quotes": [
    "The most touching moments, strong emotional expressions, user's original words"
  ]
}
```
"""

short_term_memory_prompt_only_content = """
You are an experienced memory summarizer, skilled at summarizing conversation content according to the following rules:
1. Summarize important information about the user to provide more personalized service in future conversations.
2. Do not repeat summaries, do not forget previous memories unless the total memory exceeds 1800 characters; otherwise, do not forget or compress the user's historical memory.
3. Information such as device volume changes, playing music, weather, exit, or the user's unwillingness to chat, etc.—which are not related to the user themselves—should NOT be included in the summary.
4. Data in the chat such as today's date, time, and weather that are unrelated to user events should NOT be included in the summary, as storing this information as memory may affect future conversations.
5. Do NOT include device operation results (success or failure) or meaningless user chatter in the summary.
6. Do not summarize just for the sake of summarizing; if the user's chat is meaningless, it is acceptable to just return the original history.
7. Only return the summary, strictly keeping it within 1800 characters.
8. Do NOT include code, XML, explanations, comments, or sample content—only extract information from the conversation.
"""



def extract_json_data(json_code):
    start = json_code.find("```json")
    # 从start开始找到下一个```结束
    end = json_code.find("```", start + 1)
    # print("start:", start, "end:", end)
    if start == -1 or end == -1:
        try:
            jsonData = json.loads(json_code)
            return json_code
        except Exception as e:
            print("Error:", e)
        return ""
    jsonData = json_code[start + 7 : end]
    return jsonData


TAG = __name__


class MemoryProvider(MemoryProviderBase):
    def __init__(self, config, summary_memory):
        super().__init__(config)
        self.short_memory = ""
        self.save_to_file = True
        self.memory_path = get_project_dir() + "data/.memory.yaml"
        self.load_memory(summary_memory)

    def init_memory(
        self, role_id, llm, summary_memory=None, save_to_file=True, **kwargs
    ):
        super().init_memory(role_id, llm, **kwargs)
        self.save_to_file = save_to_file
        self.load_memory(summary_memory)

    def load_memory(self, summary_memory):
        # api获取到总结记忆后直接返回
        if summary_memory or not self.save_to_file:
            self.short_memory = summary_memory
            return

        all_memory = {}
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                all_memory = yaml.safe_load(f) or {}
        if self.role_id in all_memory:
            self.short_memory = all_memory[self.role_id]

    def save_memory_to_file(self):
        all_memory = {}
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                all_memory = yaml.safe_load(f) or {}
        all_memory[self.role_id] = self.short_memory
        with open(self.memory_path, "w", encoding="utf-8") as f:
            yaml.dump(all_memory, f, allow_unicode=True)

    async def save_memory(self, msgs):
        # 打印使用的模型信息
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"使用记忆保存模型: {model_info}")
        api_key = getattr(self.llm, "api_key", None)
        memory_key_msg = check_model_key("记忆总结专用LLM", api_key)
        if memory_key_msg:
            logger.bind(tag=TAG).error(memory_key_msg)
        if self.llm is None:
            logger.bind(tag=TAG).error("LLM is not set for memory provider")
            return None

        if len(msgs) < 2:
            return None

        msgStr = ""
        for msg in msgs:
            if msg.role == "user":
                msgStr += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                msgStr += f"Assistant: {msg.content}\n"
        if self.short_memory and len(self.short_memory) > 0:
            msgStr += "历史记忆：\n"
            msgStr += self.short_memory

        # 当前时间
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msgStr += f"当前时间：{time_str}"

        if self.save_to_file:
            result = self.llm.response_no_stream(
                short_term_memory_prompt,
                msgStr,
                max_tokens=2000,
                temperature=0.2,
            )
            json_str = extract_json_data(result)
            try:
                json.loads(json_str)  # 检查json格式是否正确
                self.short_memory = json_str
                self.save_memory_to_file()
            except Exception as e:
                print("Error:", e)
        else:
            result = self.llm.response_no_stream(
                short_term_memory_prompt_only_content,
                msgStr,
                max_tokens=2000,
                temperature=0.2,
            )
            save_mem_local_short(self.role_id, result)
        logger.bind(tag=TAG).info(f"Save memory successful - Role: {self.role_id}")

        return self.short_memory

    async def query_memory(self, query: str) -> str:
        return self.short_memory
