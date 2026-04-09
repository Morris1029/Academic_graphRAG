import os
import subprocess
import re

# 配置文件相对路径
CONFIG_PATH = os.path.join("eval", "rag_eval", "config.yaml")

# 待评测的模型列表
MODELS_TO_EVAL = [ "doubao", "qwen3.5plus", "minimax2.5"]

def run_command():
    """
    运行评测命令：python -m eval.rag_eval.run
    """
    print(f"\n" + "="*50)
    print(f"正在运行 RAG 评估: python -m eval.rag_eval.run")
    print("="*50 + "\n")
    try:
        # 使用 subprocess.run 运行命令，并实时输出到终端
        subprocess.run(["python", "-m", "eval.rag_eval.run"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"评测运行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n用户中止运行。")
        return False
    return True

def update_answer_model(new_model):
    """
    更新 config.yaml 中的 answer_model 字段
    """
    print(f"\n>>> 正在将回答模型切换为: {new_model}")
    if not os.path.exists(CONFIG_PATH):
        print(f"错误: 找不到配置文件 {CONFIG_PATH}")
        return False
        
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式匹配并替换 answer_model 的值
        # 匹配格式如 answer_model: "deepseekv3.2" 或 answer_model: gemini
        pattern = r'(answer_model:\s*["\']?)([^"\']*)(["\']?)'
        
        def replace_func(match):
            prefix = match.group(1)
            suffix = match.group(3)
            # 保持原有的引号风格
            return f"{prefix}{new_model}{suffix}"
        
        new_content = re.sub(pattern, replace_func, content, count=1)
        
        if new_content == content:
            print("警告: 未在配置文件中找到 answer_model 字段或其值未发生变化。")
        
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"更新配置文件时出错: {e}")
        return False

def main():
    # 获取原始模型名称（可选，如果需要最后恢复）
    # 这里我们直接按照用户需求执行
    
    # 1. 运行当前配置下的初始评测
    print("开始初始评测（当前配置）...")
    if not run_command():
        print("初始评测失败或被中止。")
        return

    # 2. 依次切换到指定模型并运行
    for model in MODELS_TO_EVAL:
        if update_answer_model(model):
            if not run_command():
                print(f"模型 {model} 的评测失败或被中止，停止后续任务。")
                break
        else:
            print(f"无法切换模型为 {model}，跳过该模型。")

    print("\n" + "#"*50)
    print("所有自动化评测任务执行完毕！")
    print("#"*50)

if __name__ == "__main__":
    main()
