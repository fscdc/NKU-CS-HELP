# nobel_bron_country = nobel['born_country_code'].value_counts().to_frame()
# nobel_bron_country = nobel_bron_country[nobel_bron_country['born_country_code'] >= 10]
# # 绘制
# plt.figure(figsize=(15, 5))
# sns.barplot(x = nobel_bron_country.index , y = 'born_country_code' ,data = nobel_bron_country)
# plt.xlabel('Country Code')
# plt.ylabel('Number of Nobel winner')
# plt.title("Most Nobel Winner Countries")
# plt.show()


# # fulprize_university = nobel[(nobel['name_of_university'] == 'University of California')
# #                           | (nobel['name_of_university'] == 'Harvard University')
# #                           | (nobel['name_of_university'] == 'Massachusetts Institute of Technology (MIT)')
# #                           | (nobel['name_of_university'] == 'Stanford University')
# #                           | (nobel['name_of_university'] == 'University of Chicago')
# #                           | (nobel['name_of_university'] == 'University of Cambridge')
# #                           | (nobel['name_of_university'] == 'Columbia University')
# #                           | (nobel['name_of_university'] == 'Princeton University')]
# # university_and_category = fulprize_university.groupby('name_of_university')['category'].value_counts().to_frame()
# # university_and_category.columns = ['number of prize']
# # university_and_category.reset_index(inplace= True)

# # plt.figure(figsize=(30, 8))
# # sns.set(style="whitegrid")
# # sns.barplot(y = 'name_of_university' , x = 'number of prize' , hue = 'category' , dodge = False ,data = university_and_category ,palette = sns.color_palette('Paired'))
# # plt.xlabel('Number of Prizes', fontsize=14)
# # plt.ylabel('University', fontsize=14)
# # plt.title("Top Universities and Distribution of Nobel Prize Categories", fontsize=16)
# # plt.xticks(fontsize=12)
# # plt.yticks(fontsize=12)
# # plt.legend(title='Category', title_fontsize='13', fontsize='12')
# # plt.savefig('Top Universities and Distribution of Nobel Prize Categories.pdf')


# 导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None

nobel = pd.read_csv('nobel_final.csv')
print(nobel.head())


