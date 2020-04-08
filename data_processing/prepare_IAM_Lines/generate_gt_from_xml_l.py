import prep_iam_writer_map
import load_set

def prep_data_set(word_gts, in_set):
    data_set = []
    for k, v in word_gts.items():

        base_name = k
        split_base_name = base_name.split('-')

        folder_and_base = "{}/{}-{}/{}".format(split_base_name[0], split_base_name[0], split_base_name[1], base_name)
        image_file = "{}.png".format(folder_and_base)

        compare_line = "-".join(base_name.split("-")[:3])
        if not compare_line in in_set:
            continue

        data_set.append({
            "gt": v['gt'],
            "image_path": image_file,
            "err": v['err']
        })
    return data_set

def get_gt(in_set):
    author_mapping, lines_gts, word_gts = prep_iam_writer_map.get_mapping('xml')
    data_set = prep_data_set(lines_gts, in_set)
    return data_set

if "__main__" == __name__:

    training_set, val1_set, val2_set, test_set = load_set.load()
    data_set = get_gt(training_set)
    print(len(data_set))
