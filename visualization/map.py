import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Map

data = pd.read_excel('./map.xlsx', index_col=0)

c = (
    Map(init_opts=opts.InitOpts(width="1400px", height='600px'))  # 图表大小
    # 添加数据系列名称, 数据(list格式), 地图名称, 不显示小红点
    .add("Country", [list(z) for z in zip(data['国家_re'], data['paper'])], "world", is_map_symbol_show=False, is_roam=True)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 标签不显示(国家名称不显示)
    .set_global_opts(

        title_opts=opts.TitleOpts(title="2022 Biofabrication Papers Ranking"),   # 主标题与副标题名称
        visualmap_opts=opts.VisualMapOpts(max_=300, is_piecewise=True),               # 注意此max的值 所有数据的颜色都根据值来渐变
    )
)

# # 自定义设置颜色的方法
# c.set_global_opts(
#     visualmap_opts=opts.VisualMapOpts(
#         is_show=True,
#         min_=50,
#         max_=200,
#         range_color=['green', 'yellow', 'red'])
# )

c.render("2022 Biofabrication Papers Ranking.html")
