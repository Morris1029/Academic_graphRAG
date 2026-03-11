import json_repair
from  utils import call_llm_api

try:
    from config import get_config
except ImportError:
    get_config = None

class GraphQ:
    def __init__(self, dataset_name, config=None):
        if config is None and get_config is not None:
            try:
                self.config = get_config()
            except:
                self.config = None
        else:
            self.config = config
        self.llm_client = call_llm_api.LLMCompletionCall()
        self.dataset_name = dataset_name
            
    def read_schema(self, schema_path: str) -> str:
        # Always read schema as UTF-8 to avoid locale-dependent decode failures.
        with open(schema_path, "r", encoding="utf-8", errors="replace") as f:
            schema = f.read()
        return schema

    def prompt_format(self, schema: str, question: str) -> str:
        # 尝试从 Config 文件加载 Prompt (这是主路径)
        if self.config:
            # 如果你有特定的中文数据集名称，可以在这里加判断
            return self.config.get_prompt_formatted("decomposition", "general", ontology=schema, question=question)

        # 兜底 Prompt (Fallback) - 如果 Config 加载失败，使用这里的代码
        # 我们将其修改为【学术通用版】，适配你的新 Schema
        else:
            return f"""
            你是一个专业的学术问题分解专家，擅长处理多跳推理和综述类问题。
            请根据以下【图谱 Schema】和【用户问题】，将复杂问题分解为 2-4 个具体的子问题。

            【关键要求】：
            1. **子问题独立性**：每个子问题应能独立检索，专注于特定的实体（论文/模型/作者）或关系。
            2. **学术相关性**：子问题应涉及 Schema 中定义的类型（如：模型的效果、论文的发表时间、方法的局限性）。
            3. **简单问题**：如果问题是简单的（如“X的作者是谁”），直接返回原问题作为唯一的子问题。
            4. **格式**：只返回一个 JSON 对象。

            【图谱 Schema】：
            {schema}

            【用户问题】：{question}

            【示例 1 - 复杂对比】：
            用户问题："对比 Transformer 和 LSTM 在文本分类任务上的表现及局限性。"
            输出：
            {{
              "sub_questions": [
                {{"sub-question": "Transformer 在文本分类任务上的表现效果和局限性是什么？"}},
                {{"sub-question": "LSTM 在文本分类任务上的表现效果和局限性是什么？"}},
                {{"sub-question": "比较 Transformer 和 LSTM 的各项评价指标。"}},
              ],
              "involved_types": {{
                "nodes": ["模型", "任务", "评价指标", "挑战痛点"],
                "relations": ["应用于", "达到效果", "存在局限"]
              }}
            }}

            【示例 2 - 简单查询】：
            用户问题："郭海湘发表了哪些关于突发事件的论文？"
            输出：
            {{
              "sub_questions": [
                {{"sub-question": "郭海湘撰写了哪些研究主题涉及突发事件的论文？"}}
              ],
              "involved_types": {{
                "nodes": ["学者", "论文", "研究主题"],
                "relations": ["撰写", "利用"]
              }}
            }}
            """
    
    def decompose(self, question: str, schema_path: str) -> dict:
        schema = self.read_schema(schema_path)
        prompt = self.prompt_format(schema, question)
        response = self.llm_client.call_api(prompt)
        content = json_repair.loads(response)
        
        # Ensure backward compatibility - if old format, convert to new format
        if isinstance(content, list):
            content = {
                "sub_questions": content,
                "involved_types": {
                    "nodes": [],
                    "relations": [],
                    "attributes": []
                }
            }
        
        return content  
