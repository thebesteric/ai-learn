import pdfplumber

# 使用三方模块读取 PDF 文件
with pdfplumber.open("../data/report_with_table.pdf") as pdf:
    # 提取所有文本
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    print(text[:200])  # 打印前 200 字符

    # 提取表格（自动检测）
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            print("\n表格内容：")
            for row in table:
                print(row)
