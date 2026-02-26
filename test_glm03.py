import os
import sys

from utils import logger

# 确保能导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.agents.orchestrator import GLMOrchestrator
from config import get_config


def main():
    # 1. 加载配置
    print("⏳ Loading Config...")
    cfg = get_config()
    # 强制设置当前要测试的数据集为 demo
    # cfg.active_dataset = "demo"
    active_dataset = cfg.active_dataset  # "demo"
    dataset_cfg = cfg.get_dataset_config(active_dataset)

    # 2. 【关键步骤】读取 Schema 内容
    schema_path = dataset_cfg.schema_path
    print(f"📖 Loading Schema from {schema_path}...")
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_text = f.read()
    except Exception as e:
        logger.warning(f"Schema load failed: {e}. Using default.")
        schema_text = "Entities: Person, Event, Organization. Relations: participates_in, related_to."

    print("🤖 Initializing GLM Orchestrator (This loads the Graph Index)...")
    # 初始化
    bot = GLMOrchestrator(schema_text=schema_text)
    print("✅ System Ready!")

    # 针对你的问题的测试
    target_question = "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?"
    print(f"\n🧪 Auto-Testing Target Question: {target_question}")
    print("=" * 50)

    response = bot.run(target_question)

    print("\n" + "=" * 50)
    print("🏁 Final Answer:")
    print(response)
    print("=" * 50)

    # 进入交互模式
    while True:
        q = input("\nNext Question (or 'exit'): ")
        if q.lower() == 'exit': break
        print(bot.run(q))


if __name__ == "__main__":
    main()