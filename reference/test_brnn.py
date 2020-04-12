from hwr_utils.utils import *
#from torchvision.models import resnet
from models.basic import GeneralizedBRNN

if __name__ == "__main__":
    batch = 2
    y = torch.rand(batch, 256, 7, 451)
    y = y.permute(1,0,2,3)
    input_size = y.shape[2] * y.shape[3]
    rnn = GeneralizedBRNN(input_size, 200, input_size)

    output = rnn(y)
    output = output.permute(1, 0, 2, 3)
    print(output.shape)
    # final = torch.cat([a, new], dim=2)
