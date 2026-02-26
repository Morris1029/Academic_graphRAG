import pandas as pd
import os
import sys
import csv
import chardet
from datetime import datetime


def get_csv_header_safely(file_path, max_sample_lines=100):
    """
    安全地获取CSV文件的表头和前几行
    """
    print(f"正在处理文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / (1024 ** 3):.2f} GB")
    print("=" * 60)

    # 1. 检测编码
    encoding = detect_encoding(file_path)

    # 2. 读取表头和前100行
    print(f"\n尝试使用编码: {encoding}")

    try:
        # 方法1: 使用csv模块（最安全）
        print("使用csv模块读取...")
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            csv_reader = csv.reader(f)

            # 读取表头
            try:
                header = next(csv_reader)
                print(f"✓ 找到表头: {len(header)} 个字段")
            except StopIteration:
                print("✗ 文件为空")
                return None, pd.DataFrame()

            # 读取前100行数据
            data_rows = []
            for i, row in enumerate(csv_reader):
                if i >= max_sample_lines - 1:
                    break
                data_rows.append(row)

            print(f"✓ 读取了 {len(data_rows)} 行数据")

            # 创建DataFrame
            df = pd.DataFrame(data_rows, columns=header)

            return header, df

    except Exception as e:
        print(f"csv模块读取失败: {e}")
        print("\n尝试使用pandas...")

        # 方法2: 使用pandas
        try:
            df = pd.read_csv(
                file_path,
                nrows=max_sample_lines,
                encoding=encoding,
                on_bad_lines='skip',
                engine='python',
                encoding_errors='ignore'
            )
            print(f"✓ pandas读取成功: {len(df)} 行")
            return df.columns.tolist(), df
        except Exception as e2:
            print(f"pandas也失败: {e2}")

            # 方法3: 尝试不同分隔符
            return try_alternative_readers(file_path, max_sample_lines)


def detect_encoding(file_path, sample_size=10000):
    """检测文件编码"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)

        encoding = result['encoding']
        confidence = result['confidence']

        if confidence > 0.7:
            print(f"检测到编码: {encoding} (置信度: {confidence:.1%})")
            return encoding

    except Exception as e:
        print(f"编码检测失败: {e}")

    # 尝试常见编码
    common_encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'gb18030', 'cp1252']

    for enc in common_encodings:
        try:
            with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                f.readline()  # 测试读取一行
            print(f"编码 {enc} 可用")
            return enc
        except:
            continue

    print("使用默认编码: utf-8")
    return 'utf-8'


def try_alternative_readers(file_path, max_lines=100):
    """尝试不同的读取方法"""
    print("\n尝试不同的分隔符...")

    delimiters = [',', ';', '\t', '|', ' ', '\x01']  # 常见分隔符

    for delimiter in delimiters:
        try:
            print(f"尝试分隔符: {repr(delimiter)}")

            with open(file_path, 'r', encoding='latin1', errors='ignore') as f:
                # 读取第一行判断
                first_line = f.readline()
                if delimiter in first_line:
                    print(f"✓ 找到分隔符: {repr(delimiter)}")

                    # 回到文件开头
                    f.seek(0)

                    # 读取数据
                    data = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        row = line.strip().split(delimiter)
                        data.append(row)

                    if data:
                        header = data[0]
                        rows = data[1:] if len(data) > 1 else []
                        df = pd.DataFrame(rows, columns=header)
                        return header, df

        except Exception as e:
            continue

    print("所有方法都失败")
    return None, pd.DataFrame()


def display_data_info(header, df):
    """显示数据信息"""
    if header is None or df.empty:
        print("无法读取数据")
        return

    print("\n" + "=" * 60)
    print("数据信息摘要")
    print("=" * 60)

    print(f"\n1. 字段列表 (共 {len(header)} 个):")
    for i, col in enumerate(header, 1):
        print(f"  {i:3d}. {col}")

    print(f"\n2. 数据形状: {len(df)} 行 × {len(df.columns)} 列")

    print(f"\n3. 前10行数据:")
    print("-" * 60)

    # 设置显示选项
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    print(df.head(10))

    # 重置选项
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

    print(f"\n4. 数据类型:")
    print(df.dtypes.to_string())

    print(f"\n5. 内存使用: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


def estimate_file_stats(file_path):
    """估算文件统计信息（修复版本）"""
    print("\n" + "=" * 60)
    print("文件统计信息")
    print("=" * 60)

    try:
        # 文件基本信息
        file_stats = os.stat(file_path)
        file_size_gb = file_stats.st_size / (1024 ** 3)
        mod_time = datetime.fromtimestamp(file_stats.st_mtime)

        print(f"文件大小: {file_stats.st_size:,} 字节 ({file_size_gb:.2f} GB)")
        print(f"最后修改: {mod_time}")

        # 估算行数（使用单独的文件读取）
        print("\n正在估算文件行数...")
        line_count = 0
        sample_size = 0

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # 读取前1000行来估算平均行长
            lines_to_sample = 1000
            char_count = 0

            for i, line in enumerate(f):
                if i >= lines_to_sample:
                    break
                char_count += len(line)
                line_count += 1

            if line_count > 0:
                avg_line_len = char_count / line_count
                estimated_lines = int(file_stats.st_size / avg_line_len)

                print(f"采样 {line_count} 行，平均行长: {avg_line_len:.0f} 字符")
                print(f"估算总行数: {estimated_lines:,}")
                print(f"估算完整文件大小: {estimated_lines * avg_line_len / 1024 ** 3:.2f} GB")

        return {
            'file_size_gb': file_size_gb,
            'estimated_lines': estimated_lines if 'estimated_lines' in locals() else None
        }

    except Exception as e:
        print(f"统计信息获取失败: {e}")
        return {}


def save_sample_data(df, original_path, sample_size=100):
    """保存样本数据"""
    if df.empty:
        print("没有数据可保存")
        return

    # 确保不超过实际行数
    save_df = df.head(min(sample_size, len(df)))

    # 生成输出文件名
    base_name = os.path.splitext(original_path)[0]
    output_path = f"{base_name}_sample_{len(save_df)}.csv"

    try:
        save_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ 已保存 {len(save_df)} 行数据到: {output_path}")
        return output_path
    except Exception as e:
        print(f"保存失败: {e}")
        return None


def main():
    """主函数"""
    # 设置您的文件路径
    file_path = r"E:\mydata\graphRAG\271万+学术论文数据集 (2007-2025.4).csv"  # 修改这里

    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        print("\n请修改 file_path 变量为正确的文件路径")
        return

    # 1. 获取表头和前100行
    header, df = get_csv_header_safely(file_path, max_sample_lines=100)

    # 2. 显示数据信息
    display_data_info(header, df)

    # 3. 获取文件统计信息
    stats = estimate_file_stats(file_path)

    # 4. 询问是否保存
    if not df.empty:
        print("\n" + "=" * 60)
        save = input("是否保存前100行数据到新文件? (y/n): ").strip().lower()
        if save == 'y':
            saved_path = save_sample_data(df, file_path)
            if saved_path:
                print(f"样本文件: {saved_path}")
                print(f"文件大小: {os.path.getsize(saved_path) / 1024:.1f} KB")



if __name__ == "__main__":
    df = main()