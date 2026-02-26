import os
import sys

# 确保能导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.agents.orchestrator import GLMOrchestrator
from config import get_config


def main():
    # 1. 准备 Schema (模拟读取，或者从 kt_gen 读取)
    # 在真实项目中，应该从 config.dataset.schema_path 读取
    schema_text = """
    Entities: Paper, Author, Concept, Venue
    Relations: 
    - Paper --CITES--> Paper
    - Paper --AUTHORED_BY--> Author
    - Paper --PROPOSES--> Concept
    attributes: publication_year, citation_count, abstract
    """

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