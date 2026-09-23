"""Markdown 表格解析。"""
import re
import pandas as pd

def pd_read_markdown_table(
        md_table: str, missing_text: str = ""
    ) -> pd.DataFrame:

    # 清洗Markdown格式标记（仅去除样式，保留文本/空值）
    def clean_text(text: str) -> str:
        text = str(text)
        # 去除 **加粗**
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        # 去除 *斜体*
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        # 去除 `代码`
        text = re.sub(r'`(.*?)`', r'\1', text)
        # 去除 [链接](url)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # 仅去除首尾空格，保留空字符串（不删除空单元格）
        text = text.strip()
        if text == "":
            return missing_text
        return text

    # 按行拆分，保留所有行（包括空行，后续过滤）
    lines = [line.strip() for line in md_table.strip().split('\n')]
    # 过滤完全空的行
    lines = [line for line in lines if line]

    # 提取有效行：跳过表格分隔行（|---|---|）
    data_rows = []
    for line in lines:
        if re.fullmatch(r'\|?.*\---.*\|?', line):
            continue
        data_rows.append(line)

    if not data_rows:
        raise ValueError("未检测到有效的Markdown表格数据")

    # 核心修改：解析行 + 保留空单元格
    def parse_row(row: str):
        # 按 | 分割单元格
        cells = row.split('|')
        # 去掉首尾的空元素（因为Markdown表格首尾是|，分割后会产生空值）
        cells = cells[1:-1]
        # 清洗每个单元格，保留空字符串
        return [clean_text(cell) for cell in cells]

    # 解析表头和数据
    header = parse_row(data_rows[0])
    rows = [parse_row(row) for row in data_rows[1:]]

    # 生成DataFrame（严格保留空单元格）
    df = pd.DataFrame(rows, columns=header)

    return df

