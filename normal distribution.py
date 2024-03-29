# 判断数据是否服从正态分布
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 更改需要验证的数据列名
data = pd.read_csv(r'2.csv', names=['Length(μm)'])
data = data.iloc[1:, :]
data["Length(μm)"] = pd.to_numeric(data["Length(μm)"], errors='coerce')

fig = plt.figure(figsize = (10,6))

ax2 = fig.add_subplot(1,1,1)
data.hist(bins=50, ax = ax2)
data.plot(kind ='kde', secondary_y=True, ax = ax2)
plt.grid()

plt.savefig('2-normal.svg')
plt.show()
