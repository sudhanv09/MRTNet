#!/usr/bin/env python3

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm

fixed_size = 100, 100
num_image = 200
src_path = "../../Dataset/monkeys/"
dest_path = "../../Dataset/train/"
save_path = "../../Dataset/synth/"

try:
    Path(save_path).mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print("Save dir not available")
else:
    print("Save dir created")


dest_image_path = list(Path(dest_path).glob("*.jpg"))
src_image_path = list(Path(src_path).glob("*.jpg"))


for i in tqdm(range(num_image)):
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
print(f"Generated {num_image} at {Path(save_path)}")
