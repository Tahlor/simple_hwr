from PIL import Image
import numpy as np
from scipy import ndimage
import cv2

def find_center_of_mass(input_data):
    img = input_data['img']
    profile = np.sum(255 - img, axis=1)
    center_of_mass = ndimage.measurements.center_of_mass(profile)[0]
    input_data['center'] = center_of_mass
    return input_data

def pad_to_height(input_data, input_height=None):
    img = input_data['img']
    center = input_data['center']
    idx = int(center - input_height / 2)

    new_size = (img.shape[1], input_height)
    new_pil_img = Image.new("RGB", new_size, 'white')
    old_pil_img = Image.fromarray(img)
    new_pil_img.paste(old_pil_img, (0, -idx))
    new_img = np.array(new_pil_img, np.uint8)
    img = new_img

    input_data['img'] = img
    return input_data

def scale_by_author_avg_std(input_data, scale_mu=None, baseline_height=None):
    img = input_data['img']

    author_avg_std = input_data['author_avg_std']
    avg_line = author_avg_std / scale_mu
    hpercent = baseline_height / avg_line

    if img.shape[0] > 6 and img.shape[1] > 6:
        img = cv2.resize(img,None,fx=hpercent, fy=hpercent, interpolation = cv2.INTER_CUBIC)

    input_data['img'] = img
    return input_data

def apply_grayscale(input_data):
    img = input_data['img']
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:,:,None]
    input_data['img'] = img
    return input_data

def side_padding(input_data, padding=None):
    img = input_data['img']
    img = np.pad(img, ((0,0),(padding, padding),(0,0)), 'constant', constant_values=255)
    input_data['img'] = img
    return input_data

def scale_to_fixed_height(input_data, height=None, scale_width=True):
    img = input_data['img']
    percent = float(height) / img.shape[0]

    w_percent = percent
    if not scale_width:
        w_percent = 1.0

    img = cv2.resize(img, (0,0), fx=w_percent, fy=percent, interpolation = cv2.INTER_CUBIC)
    input_data['img'] = img
    return input_data

def uniform_resize(input_data, percent=None):
    img = input_data['img']
    img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
    input_data['img'] = img
    return input_data

def invert_img(input_data):
    img = input_data['img']
    img = 255 - img
    input_data['img'] = img
    return input_data