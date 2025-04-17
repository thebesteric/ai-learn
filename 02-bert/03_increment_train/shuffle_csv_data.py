import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('../hotel-intents.csv')

# 提取表头（第一行）
header = df.columns.tolist()

# 提取除第一行外的数据
data = df[1:]

# 打乱数据
shuffled_data = data.sample(frac=1).reset_index(drop=True)

# 合并表头和打乱后的数据
new_df = pd.concat([pd.DataFrame([header], columns=header), shuffled_data])

# 保存打乱后的数据到新的 CSV 文件
new_df.to_csv('../shuffled-hotel-intents.csv', index=False, header=False)