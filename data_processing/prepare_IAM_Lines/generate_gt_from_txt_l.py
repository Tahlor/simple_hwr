
import os
import load_set


def prep_data_set(words, in_set):

    data_set = []
    for i, word in enumerate(words):
        split_line = word.split(' ', 8)
        transcription = split_line[-1].replace("|", " ").strip()

        base_name = os.path.splitext(split_line[0])[0]
        split_base_name = base_name.split('-')

        folder_and_base = "{}/{}-{}/{}".format(split_base_name[0], split_base_name[0], split_base_name[1], base_name)
        image_file = "{}.png".format(folder_and_base)

        compare_line = "-".join(split_line[0].strip().split("-")[:3])
        if not compare_line in in_set:
            continue

        err_str = split_line[1]
        err = False if err_str == 'ok' else True

        data_set.append({
            "gt": transcription,
            "image_path": image_file,
            "err": err
        })

    return data_set

def get_gt(in_set):
    with open('lines.txt') as f:
        words = [s for s in f.readlines()]
    words = [s.strip() for s in words if not s.startswith('#')]
    data_set = prep_data_set(words, in_set)
    return data_set

if "__main__" == __name__:
    training_set, val1_set, val2_set, test_set = load_set.load()
    data_set = get_gt(training_set)
    print(len(data_set))
