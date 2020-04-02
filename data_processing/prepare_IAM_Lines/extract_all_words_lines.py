import generate_gt_from_txt_l
import generate_gt_from_xml_l

#We've shown words are identical for txt and xml so don't do both
import generate_gt_from_txt_w

import load_set
import json
import os
from tqdm import tqdm

if __name__ == "__main__":
    sets = load_set.load()
    set_names = ['training', 'val1', 'val2', 'test']

    generators = [generate_gt_from_txt_l, generate_gt_from_xml_l, generate_gt_from_txt_w]
    gen_paths = ['lines/txt', 'lines/xml', 'words']

    for s_name, s in tqdm(zip(set_names, sets)):
        for g_path, g in zip(gen_paths, generators):
            fullpath = os.path.join("raw_gts", g_path, s_name+'.json')
            try:
                os.makedirs(os.path.dirname(fullpath))
            except:
                pass

            with open(fullpath, 'w') as f:
                json.dump(g.get_gt(s), f)
            print(fullpath)
