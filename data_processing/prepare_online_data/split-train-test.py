import json
import random

with open('online_augmentation.json') as fp:
    data = json.load(fp)

split = .7
k = int(split * len(data))

train_indices = random.sample(range(len(data)), k)
train_data = [data[i] for i in range(len(data)) if i in train_indices]
test_data = [data[i] for i in range(len(data)) if i not in train_indices]

with open('train_augmentation.json', 'w') as fp:
    json.dump(train_data, fp, indent=2)

with open('test_augmentation.json', 'w') as fp:
    json.dump(test_data, fp, indent=2)

