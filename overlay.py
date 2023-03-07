"""
Masked wordcloud
================
Using a mask you can generate wordclouds in arbitrary shapes.
"""

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS
import json
from pathlib import Path
import jieba

# read ocr.json
with open("ocr.json", encoding="GBK") as f:
    data = json.load(f)
data = [" ".join(i) for i in data]
data = " ".join(data)
data = " ".join(jieba.cut(data))

which_image = "wolf_mask_6300.png"
# grey scale wolf mask
mask = np.array(Image.open(Path("public", which_image)))
# fill the transparent part with white
mask[mask == 0] = 255

# make all pixel above 200 white
mask[mask > 128] = 255
# make all pixel below 200 black
mask[mask < 128] = 0


stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(
    background_color="white",
    max_words=2000,
    mask=mask,
    stopwords=stopwords,
    max_font_size=2**10,
    contour_width=2,
    contour_color="black",
    color_func=lambda *args, **kwargs: "black",
    font_path=Path("public", "font", "TingFengSuShuo-2.ttf").__str__(),
)

# generate word cloud
wc.generate(data)

# store to file
wc.to_file(which_image)

# show
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.imshow(mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.axis("off")
plt.show()
