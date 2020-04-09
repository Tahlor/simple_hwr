import numpy as np

if False:
    preds_combo = [[0, 1, 1, 11],
                   [4, 5, 0, 7],
                   [8, 9, 0, 3],
                   [20, 21, 1, 23],
                   [12, 13, 1, 15],
                   [16, 17, 0, 19],
                   [32, 33, 1, 27],
                   [28, 29, 0, 31],
                   [24, 25, 0, 35]]

    preds_combo = np.asarray(preds_combo)
    k = []
    preds_combo[:,2] = np.cumsum(preds_combo[:,2])

    for i in range(0,32):
        k.append({"gt": np.asarray(preds_combo), "image_path":f"{i}"})

    np.save("training_dataset.npy", k)

    gt = np.array(range(36)).reshape(9, 4).astype(np.float64)
    gt[:, 2] = np.cumsum([1, 0, 0, 1, 0, 1, 1, 0, 0])


## Make sure these aren't the same data
import numpy as np
x = np.load("training_dataset.npy", allow_pickle=True)
y = np.load("/home/taylor/github/simple_hwr/RESULTS/pretrained/adapted_v2/training_dataset.npy", allow_pickle=True)

d, d2 = {}, {}
for i in y:
	d2[i["image_path"]] = i["gt"]

for i in x:
	d[i["image_path"]] = i["gt"]

for i in d:
	np.testing.assert_allclose(d[i][:5], d2[i][:5])

print(d[i][:20])
print(d2[i][:20])

