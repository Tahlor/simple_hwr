import sys
import json
import os
from collections import defaultdict

def load_char_set(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].iteritems():
        idx_to_char[int(k)] = v

    return idx_to_char, char_set['char_to_idx']


def make_char_set(paths, root="./data"):
    out_char_to_idx = {}
    out_idx_to_char = {}
    char_freq = defaultdict(int)

    for data_file in paths:
        with open(os.path.join(root, data_file)) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [item for key,item in data.items()]

        cnt = 1  # this is important that this starts at 1 not 0
        for data_item in data:
            for c in data_item.get('gt', ""):
                if c not in out_char_to_idx:
                    out_char_to_idx[c] = cnt
                    out_idx_to_char[cnt] = c
                    cnt += 1
                char_freq[c] += 1

    out_char_to_idx2 = {}
    out_idx_to_char2 = {}

    for i, c in enumerate(sorted(out_char_to_idx.keys())):
        out_char_to_idx2[c] = i + 1
        out_idx_to_char2[i + 1] = c

    # Add empty
    out_char_to_idx2["|"] = 0
    out_idx_to_char2[0] = "|"

    return out_char_to_idx2, out_idx_to_char2, char_freq


if __name__ == "__main__":
    character_set_path = sys.argv[-1]
    paths = [sys.argv[i] for i in range(1, len(sys.argv)-1)]

    char_to_idx, idx_to_char, char_freq = make_char_set(*paths)

    output_data = {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }

    for k,v in sorted(char_freq.iteritems(), key=lambda x: x[1]):
        print(k, v)

    print("Size:", len(output_data['char_to_idx']))

    with open(character_set_path, 'w') as outfile:
        json.dump(output_data, outfile)
