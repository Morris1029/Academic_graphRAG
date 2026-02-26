import sys
import subprocess
import shutil
import importlib
import os
from pathlib import Path


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"📊 {title}")
    print(f"{'=' * 60}")


def print_result(item, status, details=""):
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {item}: {details}")


def check_python_environment():
    print_header("Python环境检查")

    # 检查是否在虚拟环境中
    is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print_result("虚拟环境", is_venv,
                 f"真实路径: {sys.prefix}" if is_venv else "未在虚拟环境中运行")

    # 检查Python版本
    python_version = sys.version.split()[0]
    print_result("Python版本", True, python_version)

    # 检查pip版本
    try:
        pip_version = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                     capture_output=True, text=True).stdout.split()[1]
        print_result("Pip版本", True, pip_version)
    except:
        print_result("Pip版本", False, "无法获取")


def check_system_dependencies():
    print_header("系统级依赖检查")

    # 检查Java
    java_path = shutil.which('java')
    if java_path:
        try:
            java_version = subprocess.run(['java', '-version'], capture_output=True,
                                          text=True, stderr=subprocess.STDOUT)
            version_line = java_version.stderr.split('\n')[0] if java_version.stderr else "未知"
            print_result("Java", True, f"{version_line} | 路径: {java_path}")
        except:
            print_result("Java", True, f"路径: {java_path} (版本检查失败)")
    else:
        print_result("Java", False, "未安装")

    # 检查antiword
    antiword_path = shutil.which('antiword')
    if antiword_path:
        try:
            antiword_version = subprocess.run(['antiword', '-v'], capture_output=True,
                                              text=True).stdout.strip()
            print_result("Antiword", True, f"{antiword_version} | 路径: {antiword_path}")
        except:
            print_result("Antiword", True, f"路径: {antiword_path} (版本检查失败)")
    else:
        print_result("Antiword", False, "未安装")

    # 检查LibreOffice (可选)
    libreoffice_path = shutil.which('soffice') or shutil.which('libreoffice')
    print_result("LibreOffice", bool(libreoffice_path),
                 f"路径: {libreoffice_path}" if libreoffice_path else "未安装(可选)")


def check_python_packages():
    print_header("Python包依赖检查")

    # 读取requirements.txt
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        print("❌ requirements.txt 文件不存在")
        return

    # 检查每个包
    for req in requirements:
        # 解析包名（去除版本号）
        package_name = req.split('>')[0].split('<')[0].split('=')[0].split('~')[0].strip()

        try:
            # 尝试导入包
            module = importlib.import_module(package_name.replace('-', '_'))
            version = getattr(module, '__version__', '未知版本')
            print_result(package_name, True, f"版本: {version}")
        except ImportError:
            print_result(package_name, False, "未安装")


def check_spacy_models():
    print_header("spaCy模型检查")

    try:
        import spacy
        print_result("spaCy包", True, f"版本: {spacy.__version__}")

        # 检查中文模型
        try:
            nlp = spacy.load('zh_core_web_lg')
            print_result("zh_core_web_lg", True, f"路径: {nlp.path}")
        except OSError:
            print_result("zh_core_web_lg", False, "模型未安装")

        # 检查英文模型作为备选
        try:
            nlp = spacy.load('en_core_web_lg')
            print_result("en_core_web_lg", True, f"路径: {nlp.path} (备选)")
        except OSError:
            print_result("en_core_web_lg", False, "模型未安装")

    except ImportError:
        print_result("spaCy包", False, "未安装")


def check_huggingface_models():
    print_header("HuggingFace模型检查")

    try:
        from huggingface_hub import snapshot_download
        import os

        # 检查默认模型
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')

        # 检查模型是否已缓存
        model_path = None
        for root, dirs, files in os.walk(cache_dir):
            if 'all-MiniLM-L6-v2' in root:
                model_path = root
                break

        if model_path:
            print_result("HuggingFace模型", True, f"已缓存: {model_path}")
        else:
            print_result("HuggingFace模型", False, "未下载 (首次使用时会自动下载)")

    except ImportError:
        print_result("HuggingFace Hub", False, "huggingface_hub未安装")


def check_mineru_config():
    print_header("MinerU配置检查")

    # 检查Linux路径
    linux_config = "/root/magic-pdf.json"
    # 检查Windows可能的路径
    windows_configs = [
        os.path.expanduser("~/magic-pdf.json"),
        "C:/magic-pdf.json",
        "./magic-pdf.json"
    ]

    config_found = False
    config_path = ""

    # 检查所有可能的路径
    for config_file in [linux_config] + windows_configs:
        if os.path.exists(config_file):
            config_found = True
            config_path = config_file
            break

    if config_found:
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print_result("MinerU配置", True, f"路径: {config_path}")
            print(f"   配置内容: {config}")
        except:
            print_result("MinerU配置", True, f"路径: {config_path} (但格式可能有问题)")
    else:
        print_result("MinerU配置", False, "未找到配置文件")


def check_document_parsing_support():
    print_header("文档解析支持检查")

    # 检查各种文档解析库
    parsing_libraries = {
        'python-docx': 'docx文件支持',
        'striprtf': 'RTF文件支持',
        'tika': 'Apache Tika (通用解析)',
        'pdfminer': 'PDF解析',
        'PyMuPDF': 'PDF解析备选',
        'textract': '文本提取'
    }

    for lib, description in parsing_libraries.items():
        try:
            importlib.import_module(lib)
            print_result(description, True, f"{lib} 已安装")
        except ImportError:
            print_result(description, False, f"{lib} 未安装")


def main():
    print("🎯 Youtu-GraphRAG 环境完整性检查")
    print("本脚本将检查您的PyCharm虚拟环境是否满足setup_env.sh的所有要求")

    check_python_environment()
    check_system_dependencies()
    check_python_packages()
    check_spacy_models()
    check_huggingface_models()
    check_mineru_config()
    check_document_parsing_support()

    print_header("检查完成")
    print("💡 建议:")
    print("1. 所有✅项目表示满足要求")
    print("2. ❌项目需要手动安装")
    print("3. 系统级依赖(Java, antiword)需要在Windows中单独安装")
    print("4. Python包依赖应在虚拟环境中安装")


if __name__ == "__main__":
    main()