import os
import sys

from utils import logger

# 确保能导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.agents.orchestrator import GLMOrchestrator
from config import get_config


def main():
    print("⏳ Loading Config...")
    cfg = get_config()
    cfg.active_dataset = "demo"

    # 预热检索器，这步最慢，放最前面
    from models.agents.tools import get_retriever
    print("🔥 Pre-loading Index (this takes time)...")
    get_retriever()

    print("🤖 Initializing GLM Orchestrator...")
    bot = GLMOrchestrator(schema_text="...")  # Schema 此时次要了

    print("\n请输入您的问题（输入 'exit' 退出）")

    while True:
        try:
            user_input = input("\nUser Query: ").strip()

            # 检查退出条件
            if user_input.lower() == "exit":
                print("测试程序结束")
                break

            # 跳过空输入
            if not user_input:
                print("⚠️ 输入不能为空，请重新输入。")
                continue

            print(f"🧪 Testing: {user_input}")

            # 运行模型获取回答
            response = bot.run(user_input)
            print(f"\n🏁 Answer:\n{response}")

        except KeyboardInterrupt:
            # 处理 Ctrl+C 优雅退出
            print("\n\n⚠️ 中断请求，退出程序中...")
            break
        except Exception as e:
            # 处理其他可能的异常
            logger.error(f"处理问题时发生错误: {str(e)}")
            print(f"❌ 错误: {str(e)}，请重新输入问题。")


if __name__ == "__main__":
    main()