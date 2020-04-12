import sys
import os
import prep_iam_writer_map
import json
from collections import defaultdict
import cv2
import numpy as np
from scipy import ndimage


def process(data, author_mapping, prefix='words'):
    new_data = []
    all_authors = defaultdict(lambda: [])
    all_authors_baseline = defaultdict(lambda: [])
    for d in data:
        image_file = d['image_path']
        line_id = image_file.split("/")[-2]

        img = cv2.imread(os.path.join(prefix, image_file), 0)
        if img is None:
            print "There was an issue with ", image_file
            continue

        new_data.append(d)

        #during calculations, skip non-words.
        tru = d['gt']
        tru = "".join([c for c in tru if c.isalnum()])
        if len(tru) == 0:
            continue

        profile = np.sum(255 - img, axis=1)
        center_of_mass = ndimage.measurements.center_of_mass(profile)[0]

        distances = center_of_mass - np.array([float(i) for i in range(0, len(profile))])
        std = (profile * (distances ** 2.0)).sum() / profile.sum()
        std = np.sqrt(std)

        author_id, avg_line, avg_full = author_mapping[line_id]
        all_authors[author_id].append(std)
        all_authors_baseline[author_id].append(avg_line)

    all_ratios = []
    for k, v in all_authors.iteritems():
        std_mean = np.mean(v)
        line_mean = np.mean(all_authors_baseline[k])
        all_ratios.append(std_mean / line_mean)

    avg_ratio = np.mean(all_ratios)
    print "AVG RATIO:", avg_ratio

    for d in new_data:
        image_file = d['image_path']
        line_id = image_file.split("/")[-2]
        author_id, avg_line, avg_full = author_mapping[line_id]
        author_avg_std = np.mean(all_authors[author_id])
        d['author_avg_std'] = author_avg_std
        d['author_id'] = author_id

    return data

if __name__ == "__main__":

    input_file = sys.argv[1]
    prefix = sys.argv[2]
    output_file = sys.argv[3]

    author_mapping, lines_gts, word_gts = prep_iam_writer_map.get_mapping('xml')

    with open(input_file) as f:
        data = json.load(f)

    out_data = process(data, author_mapping, prefix=prefix)

    try:
        os.makedirs(os.path.dirname(output_file))
    except:
        pass

    with open(output_file, 'w') as f:
        json.dump(out_data, f)
