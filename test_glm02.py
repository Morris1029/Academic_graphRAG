import os
import sys

from utils import logger

# 确保能导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.agents.orchestrator import GLMOrchestrator
from config import get_config


def main():
    # 1. 加载配置
    cfg = get_config()
    active_dataset = cfg.active_dataset  # "demo"
    dataset_cfg = cfg.get_dataset_config(active_dataset)

    # 2. 【关键步骤】读取 Schema 内容
    schema_path = dataset_cfg.schema_path
    schema_text = ""
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            # 假设 schema 是 JSON，我们把它转成字符串
            schema_content = f.read()
            # 如果文件里已经是描述性文本更好，如果是JSON，LLM也能读懂
            schema_text = schema_content
    except Exception as e:
        logger.error(f"Failed to load schema from {schema_path}: {e}")
        # 兜底：如果没有 Schema，Agent 就真的只能瞎猜了
        schema_text = "Entities: Unknown. Relations: Unknown. (Schema loading failed)"

    # 3. 注入给 Orchestrator -> R-Agent
    bot = GLMOrchestrator(schema_text=schema_text)

    # 2. 初始化编排器
    # 注意：这会触发 tools.py 中 KTRetriever 的初始化（加载索引，比较耗时）
    print("⏳ Initializing System...")
    bot = GLMOrchestrator(schema_text=schema_text)
    print("✅ System Ready!")

    # 3. 测试用例
    test_questions = [
        "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?",  # FACTUAL
        "Compare the citation trends of GNN and Transformer from 2020 to 2024.",  # TEMPORAL
        "Give me a survey about Large Language Model Agents."  # SURVEY
    ]

    # 4. 交互式测试
    print("\n--- Interactive Mode (Type 'exit' to quit) ---")
    while True:
        try:
            user_input = input("\nUser Query: ").strip()
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue

            response = bot.run(user_input)

            print("\n" + "=" * 30)
            print("🤖 GLM Agent Answer:")
            print("=" * 30)
            print(response)
            print("=" * 30)

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()