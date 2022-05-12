import json

from pyecharts import options as opts
from pyecharts.charts import WordCloud
from PIL import Image

words = [
    ("heart", 1446),
    ("cardiac", 928),
    ("fabrication", 906),
    ("print", 825),
    ("3D", 514),
    ("bioprinting", 486),
    ("disease", 53),
    ("culture", 163),
    ("temperature", 86),
    ("hydrogel", 17),
    ("gel", 6),
    ("parameter", 1),
    ("cross-linking", 1437),
    ("pressure", 422),
    ("diameter", 353),
    ("size", 331),
    ("change", 313),
    ("control", 307),
    ("virtual", 43),
    ("silicon", 15),
    ("in-vivo", 438),
    ("control", 307),
    ("virtual", 43),
    ("silicon", 15),
    ("in-vivo", 438),
    ("control", 307),
    ("virtual", 43),
    ("silicon", 15),
    ("in-vivo", 438),
    ("cross-linking", 1437),
    ("pressure", 422),
    ("diameter", 353),
    ("size", 331),
    ("cross-linking", 1437),
    ("pressure", 422),
    ("diameter", 353),
    ("size", 331),
]


c = (
    WordCloud()
    .add("", words, word_size_range=[12, 55], mask_image='heart.jpg')
    .set_global_opts(title_opts=opts.TitleOpts(title="WordCloud for Heart Fabrication"))
    .render("wordcloud_custom_mask_image.html")
)
