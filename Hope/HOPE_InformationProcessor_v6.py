#-------------------------------------------------------------------------------
# Name:        Auto-HOPE-InformationProcessor
# Purpose:
#
# Author:      ShuoWang
#
# Created:     30/11/2021
# Copyright:   (c) ShuoWang 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import pandas as pd
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict #To make lists in order
import networkx as nx
import numpy as np


## Excel SUMIFs functions:
def sumif(df, summed_col, filtered_col, criterion):
    """sums filtered items in one column of a dataframe, and returns an integer"""
    sum= int(df.loc[df[filtered_col] == criterion][summed_col].sum())
    return(sum)

def sumif2(df, summed_col, filtered_col1, criterion1, filtered_col2, criterion2):
    """sums filtered items in two columns of a dataframe, and returns an integer"""
    sum= int(df.loc[(df[filtered_col1] == criterion1) & (df[filtered_col2] == criterion2)][summed_col].sum())
    return(sum)

def sumif3(df, summed_col, filtered_col1, criterion1, filtered_col2, criterion2,
filtered_col3, criterion3):
    """sums filtered items in three columns of a dataframe, and returns an integer"""
    sum= int(df.loc[(df[filtered_col1] == criterion1) & (df[filtered_col2] == criterion2)
    & (df[filtered_col3] == criterion3)][summed_col].sum())
    return(sum)

def sumIfContains(df, summed_col, filtered_col1, criterion1, filtered_col2, criterion2):
    """sums filtered items in two columns of a dataframe, and returns an integer"""
    sum= int(df.loc[(df[filtered_col1] == criterion1) & (df[filtered_col2] == criterion2)][summed_col].sum())
    return(sum)


def sumIntoList(df, intrst_class, criterion, num_Or_price, sys):
    """calculates number or price (based on your choice) of one interested classification and its
    percentage in one system; puts the four results in a list.
    Arguments:
        intrst_class: a string of column; num_Or_price: 'num' or 'price'; sys: a interested system. """
    if num_Or_price == 'num':
        Num_class = int(sumif2(df_sum, '数量（台套）', intrst_class, criterion, '系统', sys))
        Num_sys = int(sumif(df_sum, '数量（台套）', '系统', sys))
        numRatio_class = float("{:.2f}".format(Num_class/Num_sys*100)) #convert str of format-function back to float
        Num_non_class = int(Num_sys - Num_class)
        numRatio_non_class = 100 - numRatio_class
        list_num = [sys, Num_sys, Num_class, numRatio_class, Num_non_class, numRatio_non_class]
        return(list_num)
    elif num_Or_price == 'price':
        price_class = int(sumif2(df_sum, '总价（万元）', intrst_class, criterion, '系统', sys))
        price_sys = int(sumif(df_sum, '总价（万元）', '系统', sys))
        priceRatio_class = float("{:.2f}".format(price_class/price_sys*100))
        price_non_class = int(price_sys - price_class)
        priceRatio_non_class = 100  - priceRatio_class
        list_price = [sys, price_sys, price_class, priceRatio_class, price_non_class, priceRatio_non_class]
        return(list_price)
    else:
        print("Invalid argument. Please input 'num' or 'price' for the num_Or_price argument. ")

    list_sys=[sys, Num_sys, Price_sys]
    list_sum.append(list_sys)


def sum2IntoList(df, intrst_class1, criterion1, intrst_class2, criterion2, num_Or_price, sys):
    """calculates number or price (based on your choice) of two interested classifications and its
    percentage in one system; puts the four results in a list.
    Arguments:
        intrst_class: a string of column; num_Or_price: 'num' or 'price'; sys: an interested system. """
    if num_Or_price == 'num':
        Num_class = int(sumif3(df_sum, '数量（台套）', intrst_class1, criterion1, intrst_class2, criterion2,  '系统', sys))
        Num_sys = int(sumif(df_sum, '数量（台套）', '系统', sys))
        numRatio_class = float("{:.2f}".format(Num_class/Num_sys*100))
        Num_non_class = int(Num_sys - Num_class)
        numRatio_non_class = 100 - numRatio_class
        list_num = [sys, Num_sys, Num_class, numRatio_class, Num_non_class, numRatio_non_class]
        return(list_num)
    elif num_Or_price == 'price':
        price_class = int(sumif3(df_sum, '总价（万元）', intrst_class1, criterion1, intrst_class2, criterion2, '系统', sys))
        price_sys = int(sumif(df_sum, '总价（万元）', '系统', sys))
        priceRatio_class = float("{:.2f}".format(price_class/price_sys*100))
        price_non_class = int(price_sys - price_class)
        priceRatio_non_class = 100  - priceRatio_class
        list_price = [sys, price_sys, price_class, priceRatio_class, price_non_class, priceRatio_non_class]
        return(list_price)
    else:
        print("Invalid arugment. Please input 'num' or 'price' for the num_Or_price arugmenet. ")

    list_sys=[sys, Num_sys, Price_sys]
    list_sum.append(list_sys)


#####################################################################################################################
#####################################################################################################################

##
### 1. Read Files under a directory into one python pandas dataframe
### 1.1 Set path of to-be-processed Excel files
### Copy-n-paste the folder path in the "":
##form9Files_path= r"C:\Users\pipi\OneDrive\Work\CAS\HOPE\PythonProcessor"
##
### 1.2 Use glob to get all the Excel files in the folder and assign their paths in a list:
##path = os.getcwd()
##excel_files = glob.glob(os.path.join(form9Files_path, "*.xlsx"))
##
### 1.3. Append all Excel files together by the sequence of system names:
##df_sum = pd.DataFrame()
##
##for f in excel_files:
##    # read each file in a dataframe
##    df_each = pd.read_excel (f, engine='openpyxl') # Excel extenion 'xlrd' has explicitly
##                                                    # removed support for anything other than xls files,
##                                                    # so I used engine='openpyxl' argument.
##    df_each= df_each.drop([0]) # Delete("drop") the second row, which is normally useless
##    df_sum= df_sum.append(df_each, ignore_index = 'Ture') # Append each dataframe into the summary dataframe
##
### 1.4 Post-process dataframe:
##df_sum = df_sum[df_sum.columns.drop(list(df_sum.filter(regex='Unnamed')))] # Drop useless columns generated during appending
##
##df_sum[['系统', '子系统', '模块', '单元']] = df_sum[['系统', '子系统', '模块', '单元']].fillna(method='ffill')
### Fill the merged cells with the first row value in four specific columns
##
##df_sum.dropna(subset=['设备名称'], inplace=True) # Drop the rows where '设备名称' is NaN
##df_sum = df_sum[df_sum.设备名称 != '-'] # Drop the rows where '设备名称' is '-'
##
##df_sum['仪器类型'] = df_sum['仪器类型'].str.strip() # Drop ending spaces in "仪器类型" column

##df_sum = df_sum.replace('-', ' ', regex=True)
##df_sum['设备功率（kW）'] = df_sum['设备功率（kW）'].replace('-', np.nan, regex=True) # Replace '-' with NaN
##df_sum = df_sum.replace(chr(9), np.nan, regex=True)

# Import Summerized 表9
# sum9_excelFile_path= r"."
# sum9_excelFile = os.path.join(sum9_excelFile_path, "All_ky9_FINAL_powerUp_20220112.xlsx")
df_sum = pd.read_excel ("All_ky9_FINAL_powerUp_20220112.xlsx", engine='openpyxl')


# Import Construction design (categorized into zones) file
# area_excelFile_path= r"."
# area_excelFile = os.path.join(area_excelFile_path, "区域划分_可研_添加设备数量_20220210.xlsx")
df_area = pd.read_excel ("区域划分_可研_添加设备数量_20220210.xlsx", engine='openpyxl')

#####################################################################################################################
#####################################################################################################################

# Revisions to Table 9
### Revision on 2021/12/28: Deleted equipment by Dr.Gu:
##list_toBeDeleted = ['1.2.1.2.2', '3.3.1.2.18', '3.3.1.2.19', '3.2.3.2.2', '3.3.2.1.8', '3.2.3.2.3',
##'2.2.2.2.3', '3.2.2.1.2', '3.1.2.1.6', '1.2.1.3.5', '2.3.1.3.2', '1.1.4.1.6', '3.2.2.1.3',
##'3.2.1.3.4', '4.4.1.4.1', '4.3.4.4.1', '4.3.4.2.4', '4.2.2.1.3', '4.2.2.2.2', '4.3.2.5.1',
##'1.1.4.1.8']
### 计算共删除多少面积和钱
##df_deleted = df_sum[df_sum['序号'].isin(list_toBeDeleted)] # 得到删除的所有rows
##area_deleted = int(df_deleted['设备面积（平米）'].sum())
##price_deleted = int(df_deleted['总价（万元）'].sum())
##print('共删除'+str(area_deleted)+'平米，共删除'+str(price_deleted)+'万元。')
### 删除list里面对应的行
##df_sum = df_sum[df_sum.序号.isin(list_toBeDeleted) == False]

# 名字替reg换

# 找出名字单独改动。在area有，但在表9-final没有


#####################################################################################################################
#####################################################################################################################

# 2. 制作建筑设计表

# 按照“序号”自动填写“设备面积（平米）”和“特殊环境要求”

list_index_Area = df_area['序号'].tolist()
list_eqArea = df_area['设备名称'].tolist()
list_eqTable9 = df_sum['设备名称'].tolist()

# 比较总表9和建筑设计表的设备是否有遗漏
##list_toBeUpdate_area = []
##for i in list_eqTable9:
##    if i not in list_eqArea:
##        list_toBeUpdate_area.append(i)
##print(list_toBeUpdate_area)

# （可能不对）比较汇总表9和面积表，查漏补缺
##list_missToArea = []
##for eq in list_eqArea:
##    if eq not in list_eqTable9:
##        list_missToArea.append(df_sum.loc[df_sum['设备名称'] == eq]['序号'].item())
##
##print('以下设备没有被复制到面积汇总表中', list_missToArea)


# 按照汇总表9的序号更新面积表里的设备名称
##for index in list_index_Area:
##    df_area.loc[df_area['序号'] == index, '设备名称'] = df_sum.loc[df_sum['序号'] == index]['设备名称'].item()


# （或者）按照汇总表9的设备名称更新面积表里的序号
##for index in list_eqArea:
##    df_area.loc[df_area['设备名称'] == index]['序号'] = df_sum.loc[df_sum['设备名称'] == index]['序号'].item()

# 以df_sum的内容更新df_area的“面积”和“特殊环境”数据

##for eq in list_eqArea:
##    #设置index
##    df_area.loc[df_area['设备名称'] == eq]['序号'] = df_sum.loc[df_sum['设备名称'] == eq]['序号'].item()
##    # 同步面积
##    df_area.loc[df_area['设备名称'] == eq, '面积'] = df_sum.loc[df_sum['设备名称'] == eq]['设备面积（平米）'].item()
##    #同步特殊环境
##    df_area.loc[df_area['设备名称'] == eq, '特殊环境需求'] = df_sum.loc[df_sum['设备名称'] == eq]['特殊环境要求'].item()


# 计算各个区域面积
list_area_names = ['1', '2', '3', '4', '5', '6', '7', '8']
list_area = []
for i in range(1,9):
    df_area.loc[df_area['区域'] == i].sum()
    list_area.append(sumif(df_area, '面积', '区域', i))
dict_areaAnalysis = {'区域':list_area_names, '面积（平米）': list_area}
df_areaAnlysis = pd.DataFrame(dict_areaAnalysis)

### 每个区域内，按序号升序排列
##df_area_arranged = pd.DataFrame(list_sum, columns = ['系统', '设备总数（台/套）', '仪器总价（万元）',
##'系统总功率（kW）', '系统总面积（平米）'])
##for i in range(1,8):
##    df_area.loc[df_area['区域'] == i]


##df_area.sort_values(['区域', '序号'], inplace=True )


##df_area.to_excel('区域划分_Submited_20211231.xlsx', sheet_name='Sheet1', na_rep='',
##float_format=None, columns=None, header=True, index=False, index_label=None,
##startrow=0, startcol=0, engine='openpyxl', merge_cells=True, encoding=None,
##inf_rep='inf', verbose=True, freeze_panes=None)




#####################################################################################################################
#####################################################################################################################


# 2.Data analysis
# 2.1 Basic summary(numbers and prices) of equ in systems/subsystems/modules/apartment:

# Convert df['系统'] to a list of system names. Converts to a set to remove replicates:
list_SysNames = list(OrderedDict.fromkeys(df_sum['系统'].tolist()))

# Create a df to save numbers and sum-price for systems:
list_sum=[]
for sys in list_SysNames:
    Num_sys= sumif(df_sum, '数量（台套）', '系统', sys)
    Price_sys= sumif(df_sum, '总价（万元）', '系统', sys)
    Power_sys= sumif(df_sum, '设备功率（kW）', '系统', sys)
    Area_sys= sumif(df_sum, '设备面积（平米）', '系统', sys)
    list_sys=[sys, Num_sys, Price_sys, Power_sys, Area_sys]
    list_sum.append(list_sys)

df_SysSum = pd.DataFrame(list_sum, columns = ['系统', '设备总数（台/套）', '仪器总价（万元）', '系统总功率（kW）', '系统总面积（平米）'])

##print(df_SysSum, '\n')


# 每个系统的面积
##sumif2(df_sum, '设备面积（平米）',)

# 2.2 Key equipment summary (核心设备汇总) by looping through all systems:

# 计算整体设备数量和价格
Num_keyEq= sumif(df_sum, '数量（台套）', '是否为该系统核心设备', '是') #核心设备总台数
Price_keyEq= sumif(df_sum, '总价（万元）', '是否为该系统核心设备', '是') #核心设备总价

Num_imEq= sumif(df_sum, '数量（台套）', '是否进口', '是') #进口设备总台数
Price_imEq= sumif(df_sum, '总价（万元）', '是否进口', '是') #尽快设备总价

Num_imEq= sumif2(df_sum, '数量（台套）', '是否为该系统核心设备', '是', '是否进口', '是') #核心进口设备总台数
Price_imEq= sumif2(df_sum, '总价（万元）', '是否为该系统核心设备', '是', '是否进口', '是') #核心进口设备总价

# 统计每个系统的核心设备
list_keyEq_num = []
list_keyEq_price = []
for sys in list_SysNames:
    list_num = sumIntoList(df_sum, '是否为该系统核心设备', '是', 'num', sys)
    list_keyEq_num.append(list_num)
    list_price = sumIntoList(df_sum, '是否为该系统核心设备', '是', 'price', sys)
    list_keyEq_price.append(list_price)

df_keyEq_num = pd.DataFrame(list_keyEq_num, columns = ['系统', '设备总数量（台/套）', '核心设备数量（台/套）', '占比（%）',
'非核心设备数量（台/套）', '占比（%）'])
df_keyEq_price = pd.DataFrame(list_keyEq_price, columns = ['系统', '设备总价（万元）', '核心设备总价（万元）', '占比（%）',
'非核心设备总价（万元）', '占比（%）'])

##print(df_keyEq_num, '\n')
##print(df_keyEq_price, '\n')


# 3.3 Imported equipment summary（仪器进口汇总）

list_imEq_num = []
list_imEq_price = []
for sys in list_SysNames:
    list_num = sumIntoList(df_sum, '是否进口', '是', 'num', sys)
    list_imEq_num.append(list_num)
    list_price = sumIntoList(df_sum, '是否进口', '是', 'price', sys)
    list_imEq_price.append(list_price)

df_imEq_num = pd.DataFrame(list_imEq_num, columns = ['系统', '设备总数量（台/套）',
'进口设备数量（台/套）', '占比（%）', '非进口设备数量（台/套）', '占比（%）'])
df_imEq_price = pd.DataFrame(list_imEq_price, columns = ['系统', '设备总价（万元）',
'进口设备总价（万元）', '占比（%）', '非进口设备总价（万元）', '占比（%）'])

##print(df_imEq_num, '\n')
##print(df_imEq_price, '\n')

# 3.4 Key imported equipment summary（核心进口设备汇总）
list_keyImEq_num = []
list_keyImEq_price = []
for sys in list_SysNames:
    list_num = sum2IntoList(df_sum, '是否进口', '是', '是否为该系统核心设备', '是', 'num', sys)
    list_keyImEq_num.append(list_num)
    list_price = sum2IntoList(df_sum, '是否进口', '是', '是否为该系统核心设备', '是', 'price', sys)
    list_keyImEq_price.append(list_price)

df_keyImEq_num = pd.DataFrame(list_keyImEq_num, columns = ['系统', '设备总数量（台/套）',
'核心进口设备数量（台/套）', '占比（%）', '非核心进口设备数量（台/套）', '占比（%）'])
df_keyImEq_price = pd.DataFrame(list_keyImEq_price, columns = ['系统', '设备总价（万元）',
'核心进口设备总价（万元）', '占比（%）', '非核心进口设备总价（万元）', '占比（%）'])

##print(df_keyImEq_num, '\n')
##print(df_keyImEq_price, '\n')

# 3.5 Equipment classification summary（仪器类型汇总）

list_allCateNames = ['装备工具，不属于测试仪器','计算软硬件设备等不属于测试仪器','0101电子光学仪器','0102质谱仪器',
'0103 X射线仪器','0104光谱仪器','0105色谱仪器','0106 波谱仪器','0107电化学仪器','0108显微镜及图象分析仪器','0109热分析仪器',
'0110生化分离分析仪器','0111环境与农业分析仪器','0112样品前处理及制备仪器','011300其他','0201力学性能测试仪器','0202大地测量仪器',
'0203光电测量仪器','0204声学振动仪器','0205颗粒度测量仪器','0206探伤仪器','020700其他','0301长度计量仪器','0302热学计量仪器',
'0303力学计量仪器','0304电磁学计量仪器','0305时间频率计量仪器','0306声学计量仪器','0307光学计量仪器','030800其他',
'0401天体测量仪器','0402地面天文望远镜','0403空间天文望远镜','040400其他','0501海洋水文测量仪器','0502多要素水文气象测量系统',
'0503海洋生物调查仪器','0504海水物理量测量仪器','0505海洋遥感／遥测仪器','0506海洋采样设备','050700其他','0601电法仪器',
'0602电磁法仪器','0603磁法仪器','0604重力仪器','0605地震仪器','0606地球物理测井仪器','0607岩石矿物测试仪器','060800其他',
'0701气象台站观测仪器','0702高空气象探测仪器','0703特殊大气探测仪器','0704主动大气遥感仪器','0705被动大气遥感仪器',
'0706高层大气/电离层探测器','0707对地观测仪器','070800其他','0801通用电子测量仪器','0802射频和微波测试仪器','0803通讯测量仪器',
'0804网络分析仪器','0805大规模集成电路测试仪器','080600其他','0901临床检验分析仪器','0902影像诊断仪器','0903电子诊察仪器',
'090400其他','1001核辐射探测仪器','1002活化分析仪器','1003离子束分析仪器','1004核效应分析仪器','1005中子散射及衍射仪器',
'100600其他','1101射线检测仪器','1102超声检测仪器','1103电磁检测仪器', '1104声发射检测仪器','1105光电检测仪器','110600其他',
'120000其他仪器']

list_category_names= list(OrderedDict.fromkeys(df_sum['仪器类型'].tolist()))

### 找出“仪器类型”输入不规范的仪器的序号
##list_toBeRevised = [] # A list that stores 序号 of equipment with wrong 仪器类型
##for i in list_category_names:
##    if i not in list_allCateNames:
##        list_toBeRevised.append(df_sum.loc[df_sum['仪器类型'] == i]['序号'].tolist())
##print('“仪器类型”输入不规范的仪器的序号有' + str(list_toBeRevised))

 #统计各个仪器类型的数量和价格
list_cate_num = []
list_cate_price = []

for cat in list_category_names:
    sum_cat = int(sumif(df_sum, '数量（台套）', '仪器类型', cat))
    list_cate_num.append(sum_cat)
    sum_cat = int(sumif(df_sum, '总价（万元）', '仪器类型', cat))
    list_cate_price.append(sum_cat)

num_total = int(df_sum['数量（台套）'].sum()) # count total number of all equipment
price_total = int(df_sum['总价（万元）'].sum()) # count total price of all equipment

##print(list_cate_num)
##print(list_cate_price)

list_cat = [list_category_names, list_cate_num, list_cate_price]

# To create a df by re-arranging lists (transposed)
df_cat = pd.DataFrame(np.array(list_cat).T, columns = ['设备类型', '设备总数量（台/套）', '总价（万元）'])
df_cat = df_cat.sort_values('设备类型') #重新排序，便于展示
df_cat = df_cat.loc[df_cat['设备类型'] != 'nan'] #去掉nan行


##### 统计各个系统中，各个分类共有多少个设备，和各共计多少钱：
####
####dic_cate_num = {}
####
####
####for i in list_category:
####    list_cate_num = []
####    for sys in list_SysNames:
####        a = sumifs(df_sum, '数量（台套）','仪器类型', i, '系统', sys )
####        list_cate_num.append(a)
####    dic_cate_num[i] = list_cate_num
####
####print(dic_cate_num)


# 3.6 Self-developed equipment summary（自研率汇总）

list_selfEq_num = []
list_selfEq_price = []
for sys in list_SysNames:
    list_num = sumIntoList(df_sum, '设备来源（购置/定制/研制）', '研制', 'num', sys)
    list_selfEq_num.append(list_num)
    list_price = sumIntoList(df_sum, '设备来源（购置/定制/研制）', '研制', 'price', sys)
    list_selfEq_price.append(list_price)

df_selfEq_num = pd.DataFrame(list_selfEq_num, columns = ['系统', '设备总数量（台/套）',
'自研设备数量（台/套）', '占比（%）', '非自研设备数量（台/套）', '占比（%）'])
df_selfEq_price = pd.DataFrame(list_selfEq_price, columns = ['系统', '设备总价（万元）',
'自研设备总价（万元）', '占比（%）', '非自研设备总价（万元）', '占比（%）'])

##print(df_selfEq_num, '\n')
##print(df_selfEq_price, '\n')


# 3.7 列举所有非进口核心设备（为了查询是否是“部分零件进口”）
##df_nonImAndKeyEq = df_sum.loc[(df_sum['是否进口'] == '否') & (df_sum['是否为该系统核心设备'] == '是')]
##
##list_nonImAndKeyEq = df_nonImAndKeyEq.loc['序号'].tolist()


# 3.8 汇总所有层级的设备的数量和价格
# 子系统、模块、单元的名称汇总
list_subSys_name = list(OrderedDict.fromkeys(df_sum['子系统'].tolist()))
list_module_name = list(OrderedDict.fromkeys(df_sum['模块'].tolist()))
list_appartment_name = list(OrderedDict.fromkeys(df_sum['单元'].tolist()))

# 设备数量汇总
list_subSys_num = []
list_module_num = []
list_appartment_num = []

# 设备价格汇总
list_subSys_price = []
list_module_price = []
list_appartment_price = []

# 设备面积汇总
list_subSys_area = []
list_module_area = []
list_appartment_area = []

# 计算子系统的数量，价格和面积
for i in list_subSys_name:
    num = sumif(df_sum, '数量（台套）', '子系统', i)
    list_subSys_num.append(num)
    price = sumif(df_sum, '总价（万元）', '子系统', i)
    list_subSys_price.append(price)
    area = sumif(df_sum, '设备面积（平米）', '子系统', i)
    list_subSys_area.append(area)

# 计算模块的数量，价格和面积
for i in list_module_name:
    num = sumif(df_sum, '数量（台套）', '模块', i)
    list_module_num.append(num)
    price = sumif(df_sum, '总价（万元）', '模块', i)
    list_module_price.append(price)
    area = sumif(df_sum, '设备面积（平米）', '模块', i)
    list_module_area.append(area)

# 计算单元的数量，价格和面积
for i in list_appartment_name:
    num = sumif(df_sum, '数量（台套）', '单元', i)
    list_appartment_num.append(num)
    price = sumif(df_sum, '总价（万元）', '单元', i)
    list_appartment_price.append(price)
    area = sumif(df_sum, '设备面积（平米）', '单元', i)
    list_appartment_area.append(area)

# 汇总成三个矩阵
list_subSys = [list_subSys_name, list_subSys_num, list_subSys_price, list_subSys_area ]
list_module = [list_module_name, list_module_num, list_module_price, list_module_area]
list_apartment = [list_appartment_name, list_appartment_num, list_appartment_price, list_appartment_area]

# 形成三个df，以方便导出Excel文件
df_subSys = pd.DataFrame(np.array(list_subSys).T, columns = ['子系统', '设备总数量（台/套）', '总价（万元）', '面积（平米）'])
df_module = pd.DataFrame(np.array(list_module).T, columns = ['模块', '设备总数量（台/套）', '总价（万元）', '面积（平米）'])
df_apartment = pd.DataFrame(np.array(list_apartment).T, columns = ['单元', '设备总数量（台/套）', '总价（万元）', '面积（平米）'])


#######################################################################################################################
#######################################################################################################################
##
### 4 Plots
##
### 4.1 Tree plot
##
### convert df_fromTo to a from-to list
##
####
####df_fromTo = df_sum_noEmpt
####
##
####
####orgchart=nx.from_pandas_edgelist(edgelist,
####source='from', target='to')
####p=nx.drawing.nx_pydot.to_pydot(orgchart)
####p.write_png('orgchart.png')
##
##
##
#######################################################################################################################
#######################################################################################################################


# 5. Convert dataframes into Excel.xlsx files



# Create an Excel with all original data: 'All.xlsx'
##df_sum.to_excel('All_2021-12-28-有模块.xlsx', sheet_name='Sheet1', na_rep='',
##float_format=None, columns=None, header=True, index=False, index_label=None,
##startrow=0, startcol=0, engine='openpyxl', merge_cells=True, encoding=None,
##inf_rep='inf', verbose=True, freeze_panes=None)


# Create an Excel with all calculated results in respective sheets: 'DataAnalysis.xlsx'

# 合并设备数量和价格两个lists
df_keyEq = pd.concat([df_keyEq_num, df_keyEq_price.iloc[:, [1,2,3,4,5]]], axis=1)
df_imEq = pd.concat([df_imEq_num, df_imEq_price.iloc[:, [1,2,3,4,5]]], axis=1)
df_keyImEq = pd.concat([df_keyImEq_num, df_keyImEq_price.iloc[:, [1,2,3,4,5]]], axis=1)
df_selfEq = pd.concat([df_selfEq_num, df_selfEq_price.iloc[:, [1,2,3,4,5]]], axis=1)

# 所有需要统计的项目，为后面loop用。
# *** 顺序最好不要改变。方便以后手动更新
list_df_names = [df_SysSum, df_subSys, df_module, df_apartment, df_areaAnlysis,
                 df_keyEq, df_imEq, df_keyImEq, df_selfEq, df_cat]
list_sheet_names = ['总体概况', '子系统概况', '模块概况', '单元概况', '区域面积概况',
'核心设备', '进口设备', '核心进口设备', '自研设备','设备类别']

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('Data_Analysis_Before Qin.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
for df, sheet in zip(list_df_names, list_sheet_names):
    "Jingmin: 生成的excel表格如果不想保留第一列之前多余的计数列（即行索引） 添加参数index=False" \
        "如果不想保留每列的列名（即列索引 参考486~488设置的column的名称） 添加参数header=False"
    df.to_excel(writer, sheet_name=sheet, index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()


# Sort equipment by category
##df_sortByCat = df_sum.sort_values('仪器类型')
##df_sortByCat.to_excel('SortByCat.xlsx', sheet_name='Sheet1', na_rep='',
##float_format=None, columns=None, header=True, index=False, index_label=None,
##startrow=0, startcol=0, engine='openpyxl', merge_cells=True, encoding=None,
##inf_rep='inf', verbose=True, freeze_panes=None)






### Create an Excel with the data Mag. Qin needs: 'QinNeeds.xlsx'
### Delete all rows if '设备名称' is empty; keep the original df_sum dataframe
####df_sum_noEmpt = df_sum.dropna(subset=['设备名称'], inplace=False)
##

# 选取感兴趣的列和列的顺序，定义一个新df
##df_qinNeeds = df_sum_noEmpt[['系统', '子系统', '模块', '序号', '单元', '设备名称',
##'设备来源（购置/定制/研制）', '是否进口', '单价（万元/人民币）', '设备功率（kW）',
##'数量（台套）', '总价（万元）', '仪器类型', '是否为该系统核心设备',  '特殊环境要求', '设备面积（平米）', '设备分级(A/B/C/D)' ]]
##
##df_qinNeeds.to_excel('QinNeeds.xlsx', sheet_name='Sheet1', na_rep='',
##float_format=None, columns=None, header=True, index=True, index_label=None,
##startrow=0, startcol=0, engine='openpyxl', merge_cells=True, encoding=None,
##inf_rep='inf', verbose=True, freeze_panes=None)

### 筛选所有定制的设备
####df_dingzhi = df_sum.loc[df_sum['设备来源（购置/定制/研制）']=='定制']
####
####df_dingzhi.to_excel('所有定制设备.xlsx', sheet_name='Sheet1', na_rep='',
####float_format=None, columns=None, header=True, index=True, index_label=None,
####startrow=0, startcol=0, engine='openpyxl', merge_cells=True, encoding=None,
####inf_rep='inf', verbose=True, freeze_panes=None)
