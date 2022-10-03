import os
import cv2
import numpy as np
import sys
import random
import json

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def make_image(test_str, fonttype, fontsize, target_height):
    font = ImageFont.truetype(fonttype, fontsize)
    text_width, text_height = font.getsize(test_str)
    pd = 20

    # Baseline sits on the fontsize, I think...

    image = Image.new("RGB", (text_width+pd, text_height), (255,255,255))
    draw = ImageDraw.Draw(image)
    draw.text((pd/2, 0), test_str, (0,0,0), font=font)
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    dif = open_cv_image.shape[0] - target_height
    if dif > 0:
        open_cv_image = open_cv_image[:-dif]
    else:
        fill = np.full((target_height, open_cv_image.shape[1], open_cv_image.shape[2]), 255, np.uint8)
        fill[:open_cv_image.shape[0],:,:] = open_cv_image
        open_cv_image = fill

    return open_cv_image


if __name__ == "__main__":

    img_height = int(sys.argv[1])
    font_size = int(sys.argv[2])
    text_lines_path = sys.argv[3]
    number_of_examples = int(sys.argv[4])

    with open(text_lines_path) as f:
        text_lines = f.readlines()
    text_lines = [t.strip() for t in text_lines]
    text_lines = [t for t in text_lines if len(t) > 0]

    all_fonts = []

    for root, dirs, files in os.walk("fonts"):
        for file in files:
            if file.endswith(".ttf"):
                 all_fonts.append(os.path.join(root, file))

    output_path = "output"
    output_data =[]

    try:
        os.makedirs(output_path)
    except:
        pass

    for i, txt in enumerate(tqdm(random.sample(text_lines, number_of_examples))):
        img = make_image(txt, all_fonts[0], font_size, img_height)
        img_path = os.path.join(output_path, str(i)+".png")

        cv2.imwrite(img_path, img)
        output_data.append({
            "image_path": str(i)+".png",
            "gt": txt
        })

    train_cnt = int(number_of_examples * 0.9)
    val_cnt = number_of_examples - train_cnt

    with open("training.json", 'w') as f:
        json.dump(output_data[:train_cnt], f)

    with open("validation.json", 'w') as f:
        json.dump(output_data[train_cnt:], f)
