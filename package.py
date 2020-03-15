import random 
import numpy as np
import pandas as pd

classify_dic = {0: [], 1: ['信用卡'], 2: ['信用卡', '财富金']}

k = list(classify_dic.keys())
v = list(classify_dic.values())
df = pd.DataFrame(list(zip(k, v)), columns=['行数', '类别'])
classify_df = df.drop(columns='行数')
print(df)
print(classify_df)
print("nooooo")
print('yes')