#-------------------------------------------------------------------------------
# Name:        dfUpdate
# Purpose:     To update a dataframe based on another dataframe
#
# Author:      pipi
#
# Created:     08/01/2022
# Copyright:   (c) pipi 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import pandas as pd
from collections import OrderedDict #To make lists in order
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict #To make lists in order
import networkx as nx
import numpy as np

# 根据两个df的共同的一列作为index，按照一个正确的df来更新另一个df
def colUpdate(df_toBeUpdated, df_correct, col_toBeUpdated, col_basedOn):
    """By using "col_basedOn" column as index(comparing it in the two dataframes),
    to update "col_toBeUpdated" column of "df_toBeUpdated" dataframe according to
    "df_correct" dataframe.

    NOTE: 1. All cells in "col_basedOn" column MUST be unique (no duplicates).
    2. Two dataframes MUST have the same "col_basedOn" column (OK if in different sequences). """

    list_basedOn = df_correct[col_basedOn].tolist() # to create a list of indexes
    for i in list_basedOn:
        df_toBeUpdated.loc[df_toBeUpdated[col_basedOn] == i, col_toBeUpdated] = df_correct.loc[df_correct[col_basedOn] == i][col_toBeUpdated].item()


# 比较两个df的某一共同列，根据一个正确的df来查看另一个df是否在该列有缺少的内容
def colCheck(df_toBeUpdated, df_correct, col_toBeUpdated, col_common):
    """Comparing a common column in two dataframes, and find what's missing in
    to-be-updated one based on correct one. Return a list of missed content."""
    list_toBeUpdated = list(OrderedDict.fromkeys(df_toBeUpdated[col_common].tolist()))
    list_correct = list(OrderedDict.fromkeys(df_correct[col_common].tolist()))
    list_missed = []
    list_toBeDeleted = []
    for i in list_correct:
        if i not in list_toBeUpdated:
            list_missed.append(i)
    print("These items are missed in", col_toBeUpdated, "column:", (list_missed), ".")

    for i in list_toBeUpdated:
        if i not in list_correct:
            list_toBeDeleted.append(i)
    print("These items need to be deleted in", col_toBeUpdated, "column:",  str(list_toBeDeleted) + ".")


# Import correct 表9
sum9_excelFile_path= r"C:\Users\pipi\OneDrive\Work\CAS\HOPE\PythonProcessor"
sum9_excelFile = os.path.join(sum9_excelFile_path, "All_ky9_FINAL_powerUp_20220112.xlsx")
df_sum = pd.read_excel(sum9_excelFile, engine='openpyxl')
'''Jingmin: 另一种读入excel的方法是在第一个参数中直接输入 ‘路径+文件名’ 如下 '''
# df_sum = pd.read_excel ('All_ky9_FINAL_powerUp_20220112.xlsx', engine='openpyxl')


# Import to-be-updated table
area_excelFile_path= r"C:\Users\pipi\OneDrive\Work\CAS\HOPE\PythonProcessor"
area_excelFile = os.path.join(area_excelFile_path, "区域划分_可研_FINAL_20220120.xlsx")
df_toBeUpdated = pd.read_excel (area_excelFile, engine='openpyxl')

df_toBeUpdated["数量（台套）"] = np.nan

colUpdate(df_toBeUpdated, df_sum, '数量（台套）', '序号')

# Create an Excel with all original data: 'All.xlsx'
df_toBeUpdated.to_excel('区域划分_可研_添加设备数量_20220210.xlsx', sheet_name='Sheet1', na_rep='',
float_format=None, columns=None, header=True, index=False, index_label=None,
startrow=0, startcol=0, engine='openpyxl', merge_cells=True, encoding=None,
inf_rep='inf', verbose=True, freeze_panes=None)