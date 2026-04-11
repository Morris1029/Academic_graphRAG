import os
import subprocess
import re

# 配置文件相对路径
CONFIG_PATH = os.path.join("eval", "kg_eval", "config.yaml")

# 待评测的模型列表
MODELS_TO_EVAL = ["deepseekv3.2", "qwen3.5plus", "doubao", "kimi2.5", "glm4.7", "minimax2.5"]

def run_command():
    """
    运行评测命令：python eval/kg_eval/run.py run
    """
    cmd = ["python", "eval/kg_eval/run.py", "run"]
    print(f"\n" + "="*50)
    print(f"正在运行 KG 评估: {' '.join(cmd)}")
    print("="*50 + "\n")
    try:
        # 使用 subprocess.run 运行命令，并实时输出到终端
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"评测运行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n用户中止运行。")
        return False
    return True

def update_candidate_model(new_model):
    """
    更新 config.yaml 中的 candidate_model 字段
    """
    print(f"\n>>> 正在将候选模型切换为: {new_model}")
    if not os.path.exists(CONFIG_PATH):
        print(f"错误: 找不到配置文件 {CONFIG_PATH}")
        return False
        
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        found = False
        # 匹配格式如 candidate_model: "deepseekv3.2" 或 candidate_model: deepseekv3.2
        pattern = re.compile(r'^(\s*candidate_model:\s*)(["\']?)([^"\']*)(["\']?)(.*)$')
        
        for line in lines:
            match = pattern.match(line)
            if match:
                prefix = match.group(1)
                quote_start = match.group(2)
                # match.group(3) 是旧模型
                quote_end = match.group(4)
                suffix = match.group(5)
                # 保持原有的引号风格
                new_line = f"{prefix}{quote_start}{new_model}{quote_end}{suffix}\n"
                new_lines.append(new_line)
                found = True
            else:
                new_lines.append(line)
        
        if not found:
            print("警告: 未在配置文件中找到 candidate_model 字段。")
            return False
            
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    except Exception as e:
        print(f"更新配置文件时出错: {e}")
        return False

def main():
    print(f"开始自动化 KG 评估流程，共 {len(MODELS_TO_EVAL)} 个模型。")
    
    for i, model in enumerate(MODELS_TO_EVAL):
        print(f"\n进度: {i+1}/{len(MODELS_TO_EVAL)}")
        if update_candidate_model(model):
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
