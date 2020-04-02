from xml.etree import ElementTree as ET
import json
from tqdm import tqdm
from os import getcwd, walk
from os.path import join, normpath, basename, exists
from html import unescape
from pathlib import Path

def clean_text(s):
    return unescape(s)

def strip_rightmost_alphabetic(s):
    L = list(s)
    for i in range(len(s)):
        if L[-1].isalpha():
            L.pop()
        else:
            break
    return "".join(L)

def get_image_path_from_id(img_id, IMG_DIR):
    components = img_id.split("-")
    folder_1 = components[0]
    folder_2 = "-".join([components[0], strip_rightmost_alphabetic(components[1])])
    fname = img_id + ".tif"
    return join(IMG_DIR, folder_1, folder_2, fname)

def prepend_cwd(img_path):
    cwd = basename(normpath(getcwd()))
    return join(cwd, img_path)

def get_xml_files(xml_dir):
    xml_files = []
    for root, dirs, files in walk(xml_dir):
        xml_files += [join(root, file) for file in files]
    return xml_files

def check_uniqueness(img_json):
    # Verify uniqueness
    from collections import defaultdict
    counts = defaultdict(int)
    for item in img_json:
        counts[item["image_path"]] += 1
    for key, item in counts.items():
        if item > 1:
            print(key, item)

def get_train_test_splits(folder="./training_test_splits"):
    final_dict = {}
    split_dict = {"trainset.txt":"train", "testset_f.txt":"test", "testset_v.txt":"val1", "testset_t.txt":"val2"}
    for f in Path(folder).rglob("*.txt"):
        value = split_dict[f.name]
        for line in f.open(mode="r"):
            line = line.strip()
            assert line not in final_dict
            final_dict[line] = value
    return final_dict

def main():
    IMG_DIR = 'lineImages'
    XML_DIR = 'original-xml-all'
    img_json = []
    split_dict = get_train_test_splits()

    if not exists(IMG_DIR) or not exists(XML_DIR):
        raise Exception(f"Verify {prepend_cwd(IMG_DIR)} and {prepend_cwd(XML_DIR)} exist.")

    for file in tqdm(get_xml_files(XML_DIR)):
        root = ET.parse(file).getroot()
        transcription = root.find('Transcription')

        if not transcription:
            continue

        for line in transcription.findall('TextLine'):
            gt = clean_text(line.get('text'))
            img_id = line.get('id')

            # Correction needed
            img_id = img_id.split("-")
            f = Path(file)
            z_or_w = f.stem[-1] if f.stem[-1] in ["z","w"] else ""
            set_id = str(f.parent.name) + z_or_w
            img_id = f"{set_id}-{img_id[2]}"
            img_path = get_image_path_from_id(img_id, IMG_DIR)

            dataset = split_dict[set_id] if set_id in split_dict else "unknown"
            if exists(img_path):
                full_img_path = prepend_cwd(img_path)
                img_json.append({'gt': gt, 'image_path': full_img_path, 'augmentation': True, 'dataset':dataset, 'set_id':set_id, 'img_id':img_id})

    print(f"Found {len(img_json)} images")
    if img_json:
        with open('online_augmentation.json', 'w') as fp:
            json.dump(img_json, fp, indent=2)
    else:
        raise Exception("No images found!")
    return img_json

if __name__ == "__main__":
    img_json = main()
    check_uniqueness(img_json)
