from __future__ import print_function

import crnn
import torch

from torch.nn import CrossEntropyLoss

import matplotlib
matplotlib.use('Agg')

mlp = crnn.MLP(5, 10, [5,6,7], dropout=.8)

if torch.cuda.is_available() and False:
    mlp.cuda()
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()

print(mlp.classifier[-6])

n = mlp.classifier(torch.ones(5))
for k in mlp.classifier.modules():
    print(k)

print("THIS", mlp.classifier[5])
m = mlp.classifier[0:6](torch.ones(5))
print(m, n)
# for x in mlp.classifier.named_modules():
#     print(x)

