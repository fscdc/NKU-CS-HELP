import numpy as np
import csv

# 提取CSV文件中的数据（假设最后一列为年龄）
age = []
with open('./Experiment4/src/assets/nobel_final.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # 跳过标题行
    for row in reader:
        age.append(row[-1])

# 对数据进行分组
age = np.array(age, dtype=int)
age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_group = np.digitize(age, age_bins, right=True)

# 统计每个年龄段的人数
age_group_count = np.zeros(len(age_bins) - 1, dtype=int)
for i in range(1, len(age_bins)):
    age_group_count[i - 1] = np.sum(age_group == i)

# 打印每个年龄段的人数
for i in range(1, len(age_bins)):
    print(f"Age range from {age_bins[i-1]} to {age_bins[i]}: {age_group_count[i-1]} people")

# 最小年龄
print(f"Min age: {np.min(age)}")

# 最大年龄
print(f"Max age: {np.max(age)}")