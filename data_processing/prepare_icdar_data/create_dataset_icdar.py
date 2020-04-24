import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
from pathlib import Path

INPUT = Path("../../data/ICDAR_strokes")
OUTPUT = Path("online_coordinate_data/ICDAR")

train_data = []
test_data = []
df = pd.read_csv(INPUT / "unnormalized_data.csv")
grouped = df.groupby('signature_id')

for sig_id, group in grouped:
    info = {}
    input_img_path = (INPUT / "images" / str(sig_id).zfill(4)).with_suffix('.jpg')
    output_image_path = str(OUTPUT / f"images/{str(sig_id).zfill(4)}.jpg")

    img = plt.imread('../data' / input_img_path)

    info['full_img_path'] = output_image_path
    info['image_path'] = output_image_path
    info['xml_path'] = INPUT.as_posix()

    x_raw = group['x'].values
    y_raw = group['y'].values
    y_raw = y_raw.max() - y_raw
    factor = y_raw.max()-y_raw.min()
    x = (x_raw-x_raw.min())/factor
    y = (y_raw-y_raw.min())/factor
    # y_raw = y
    # x_raw = x
    #
    #y = y.max() - y

    # plt.plot(x,y)
    # plt.show()
    # input()

    info['x'] = x.astype('float').tolist()
    info['y'] = y.astype('float').tolist()
    info['t'] = group['time'].values.astype('int').tolist()
    info['start_times'] = [0.0]
    info['start_strokes'] = [1 if i == 0 else 0 for i in range(len(group))]
    info['x_to_y'] = float(img.shape[1]/img.shape[0])
    info['raw'] = [{'x': x_raw.astype('int').tolist(), 'y': y_raw.astype('int').tolist(), 'time': info['t']}]
    info['shape'] = list(img.shape)
    stroke_number = np.ones_like(x)
    eos = np.zeros_like(x)
    eos[-1] = 1
    info['gt'] = np.stack([x, y, stroke_number, eos]).transpose([1, 0]).astype('float').tolist()
    info['number_of_samples'] = len(x)

    if np.random.randn() < 0.95:
        info['dataset'] = 'train'
        train_data.append(info)
    else:
        info['dataset'] = 'test'
        test_data.append(info)

out = Path(f'../../data/{OUTPUT}/train_online_coords.json').resolve()
print(out)
with open(out, 'w') as fh:
    json.dump(train_data, fh)
with open(f'../../data/{OUTPUT}/test_online_coords.json', 'w') as fh:
    json.dump(test_data, fh)
