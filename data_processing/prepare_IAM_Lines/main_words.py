import os
import sys
import prep_iam_writer_map
import numpy as np
from scipy import ndimage
import cv2
from collections import defaultdict
import json

with open("task/trainset.txt") as f:
    training_set = set([s.strip() for s in f.readlines()])

with open("task/validationset1.txt") as f:
    val1_set = set([s.strip() for s in f.readlines()])

with open("task/validationset2.txt") as f:
    val2_set = set([s.strip() for s in f.readlines()])

with open("task/testset.txt") as f:
    test_set = set([s.strip() for s in f.readlines()])

author_mapping, lines_gts, word_gts = prep_iam_writer_map.get_mapping('xml')

def prep_set(data_set, lines_gts, author_mapping):
    tmp_cnt = 0

    all_authors = defaultdict(lambda: [])
    all_authors_baseline = defaultdict(lambda: [])
    for d in data_set:
        split_name = d.split('-')
        folder_and_base = "{}/{}-{}/{}".format(split_name[0], split_name[0], split_name[1], d)
        image_file = "words/{}.png".format(folder_and_base)
        line_id = "{}-{}".format(split_name[0], split_name[1])

        img = cv2.imread(image_file, 0)
        if img is None:
            print "There was an issue with ", image_file
            continue

        profile = np.sum(255 - img, axis=1)
        center_of_mass = ndimage.measurements.center_of_mass(profile)[0]

        distances = center_of_mass - np.array([float(i) for i in range(0, len(profile))])
        std = (profile * (distances ** 2.0)).sum() / profile.sum()
        std = np.sqrt(std)

        author_id, avg_line, avg_full = author_mapping[line_id]
        all_authors[author_id].append(std)
        all_authors_baseline[author_id].append(avg_line)

        if line_id == 'b06-008' or True:
            tmp_cnt += 1

    print "TMP CNT", tmp_cnt

    all_vals = []
    all_ratios = []
    iteresting_values = []
    all_lines = []

    print "Length", len(all_authors)
    # 0/0

    for k, v in all_authors.iteritems():
        std_mean = np.mean(v)
        line_mean = np.mean(all_authors_baseline[k])
        all_vals.append(std_mean)
        all_ratios.append(std_mean / line_mean)
        all_lines.append(line_mean)
    avg_ratio = np.mean(all_ratios)

    output_data = []
    for d in data_set:
        split_name = d.split('-')
        folder_and_base = "{}/{}-{}/{}".format(split_name[0], split_name[0], split_name[1], d)
        image_file = "words/{}.png".format(folder_and_base)
        line_id = "{}-{}".format(split_name[0], split_name[1])

        img = cv2.imread(image_file, 0)
        if img is None:
            print "There was an issue with ", image_file
            continue

        author_id, avg_line, avg_full = author_mapping[line_id]

        author_avg_std = np.mean(all_authors[author_id])

        v = {}
        v['image_path'] = "{}.png".format(folder_and_base)
        v['author_avg_std'] = author_avg_std
        v['gt'] = lines_gts[d]

        output_data.append(v)

    return output_data, avg_ratio

def generate_word_dataset(data_set, word_gts):
    line_to_word = defaultdict(list)
    for word_id in word_gts.keys():
        split_name = word_id.split('-')
        line_id = "{}-{}-{}".format(split_name[0], split_name[1], split_name[2])
        line_to_word[line_id].append(word_id)

    all_ids = []
    for d in data_set:
        all_ids.extend(line_to_word[d])

    return all_ids


training_set = generate_word_dataset(training_set, word_gts)
val1_set = generate_word_dataset(val1_set, word_gts)
val2_set = generate_word_dataset(val2_set, word_gts)
test_set = generate_word_dataset(test_set, word_gts)

training_output, avg_ratio = prep_set(training_set, word_gts, author_mapping)
print "Training Avg Ratio: ", avg_ratio
val1_output, _ = prep_set(val1_set, word_gts, author_mapping)
val2_output, _ = prep_set(val2_set, word_gts, author_mapping)
test_output, _ = prep_set(test_set, word_gts, author_mapping)


print len(training_output)
print len(val1_output)
print len(val2_output)
print len(test_output)

try:
    os.makedirs("output")
except:
    pass

with open("output/training.json", "w") as f:
    json.dump(training_output, f)

with open("output/val1.json", "w") as f:
    json.dump(val1_output, f)

with open("output/val2.json", "w") as f:
    json.dump(val2_output, f)

with open("output/test.json", "w") as f:
    json.dump(test_output, f)
