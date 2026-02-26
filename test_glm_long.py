import os
import sys

from utils.logger import logger

# 路径 hack
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.agents.orchestrator import GLMOrchestrator
from config import get_config


def main():
    print("⏳ Loading Config...")
    cfg = get_config()
    cfg.active_dataset = "demo"  # 确保这里和你上传的数据集一致

    # 预热检索器 (解决慢的问题：先加载模型)
    from models.agents.tools import get_retriever
    print("🔥 Pre-loading Index (Check local model path if this hangs)...")
    get_retriever()

    # 加载 Schema
    dataset_cfg = cfg.get_dataset_config(cfg.active_dataset)
    try:
        with open(dataset_cfg.schema_path, "r", encoding="utf-8") as f:
            schema_text = f.read()
    except:
        schema_text = "Standard Graph Schema"

    print("🤖 Initializing GLM Orchestrator...")
    bot = GLMOrchestrator(schema_text=schema_text)

    target_question = "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?"
    print(f"\n🧪 Running Test: {target_question}")
    print("=" * 50)

    response = bot.run(target_question)

    print("\n" + "=" * 50)
    print("🏁 Final Answer:")
    print(response)
    print("=" * 50)


if __name__ == "__main__":
    main()