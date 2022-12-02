#!/usr/bin/env python3

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


fixed_size = 100, 100
num_image = 2000
src_path = "../data/monkeys/"
dest_path = "../data/train/"
save_path = "../data/synth/"

dest_image_path = list(Path(dest_path).glob("*.jpg"))
src_image_path = list(Path(src_path).glob("*.jpg"))

log = open("pos.txt", "w+")


for i in range(num_image):
    dest_img_select = np.random.choice(dest_image_path)
    src_img_select = np.random.choice(src_image_path)

    try:
        dest_img = Image.open(dest_img_select)
        src_img = Image.open(src_img_select)
    except:
        continue

    dest_img = ImageOps.fit(dest_img, (600, 600))
    src_img = ImageOps.fit(src_img, fixed_size)

    dest_copy = dest_img.copy()
    x, y = (random.randrange(0, 500), random.randrange(0, 500))
    dest_copy.paste(src_img, (x, y))
    dest_copy.save(Path(save_path + f"a_{i}.jpg"))

    log.write(f"{src_img_select}, {x}, {y}\n")

log.close()
