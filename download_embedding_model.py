"""
Download BAAI/bge-small-zh-v1.5 embedding model from HuggingFace mirror.

Usage:
    python download_embedding_model.py

The model will be cached to HuggingFace's default cache directory
(typically ~/.cache/huggingface/hub/).
"""

import os

# 设置 HuggingFace 镜像地址（在导入任何 HF 库之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def main():
    print(f"📥 正在从 HuggingFace 镜像下载模型: {MODEL_NAME}")
    print(f"   镜像地址: {os.environ.get('HF_ENDPOINT', 'default')}")
    print()

    model = SentenceTransformer(MODEL_NAME)

    # 验证模型
    dim = model.get_sentence_embedding_dimension()
    max_seq_length = model.max_seq_length
    print(f"✅ 模型下载并加载成功!")
    print(f"   模型名称: {MODEL_NAME}")
    print(f"   输出维度: {dim}")
    print(f"   最大序列长度: {max_seq_length}")
    print()

    # 测试编码
    test_texts = [
        "知识图谱是一种结构化的知识表示方法",
        "Graph Retrieval-Augmented Generation",
    ]
    embeddings = model.encode(test_texts)
    print(f"🧪 编码测试通过:")
    for i, text in enumerate(test_texts):
        print(f"   [{i}] \"{text[:30]}...\" → shape={embeddings[i].shape}")

    print()
    print("🎉 模型已缓存到 HuggingFace 默认路径，可直接使用！")


if __name__ == "__main__":
    main()
