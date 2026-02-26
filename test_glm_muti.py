import os
import sys
import time

from utils.logger import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.agents.orchestrator import GLMOrchestrator
from config import get_config
from models.agents.tools import get_retriever


def main():
    print("⏳ Loading Config...")
    cfg = get_config()
    cfg.active_dataset = "demo"

    print("🔥 Pre-loading Engines (Vector Store & Graph)...")
    # 这一步是整个程序最耗时的地方，但只会在启动时做一次
    start = time.time()
    get_retriever()
    print(f"✅ Engines Loaded in {time.time() - start:.2f}s")

    print("🤖 Initializing Agents...")
    bot = GLMOrchestrator()  # Schema 现在由 Tool 内部处理，不需要显式传字符串

    # question = "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?"
    # question = "In which competition did Messi score a goal that was compared to Diego Maradona's goal of the century?"
    # question = "Which club offered Messi a contract after his first full season at Barcelona's youth academy?"

    print("\n--- Interactive Mode (Type 'exit' to quit) ---")
    while True:
        try:
            user_input = input("\nUser Query: ").strip()
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue
            t0 = time.time()
            response = bot.run(user_input)
            t1 = time.time()

            print("\n" + "=" * 50)
            print(f"🏁 Final Answer (Time: {t1 - t0:.2f}s):")
            print(response)
            print("=" * 50)

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()