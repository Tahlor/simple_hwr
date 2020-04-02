import torch
from ctcdecode import CTCBeamDecoder
import numpy as np

indexToCharacter = {0:"|", 1:"a", 2:"b"}
labels = ["|", "c", "c"]
my_decoder = CTCBeamDecoder(labels=labels, blank_id=0, beam_width=5, num_processes=1, log_probs_input=False)
softmax = torch.nn.Softmax(dim=2)

output = np.array([[5,10,5],[100,5,5],[5,100,5],[5,100,5],[5,100,5]]) # batch x seq x label_size; each row is the label probabilities; columns are sequence desired_num_of_strokes
output = softmax(torch.tensor(output[None, :,:]).float())

def beam(out):
    #print(out.shape)
    pred, scores, timesteps, out_seq_len = my_decoder.decode(out)
    print(f"output {pred}") # BATCHES X BEAMS X SEQ LEN
    #print(f"scores {scores}")
    #print(f"timesteps {timesteps}")
    print(f"out_seq_len {out_seq_len}")
    x = list(lookup(pred, out_seq_len, indexToCharacter))
    print(x)

def lookup(output, output_lengths, indexToCharacter):
    output = output.data.int().numpy()
    output_lengths = output_lengths.data.data.numpy()
    rank = 0 # get top ranked prediction
    # Loop through batches
    for batch in range(output.shape[0]):
        line_length = output_lengths[batch][rank]
        line = output[batch][rank][:line_length]
        string = u""
        for char in line:
            string += indexToCharacter[char]
        yield string

beam(output)


# import kenlm
# model = kenlm.Model('lm/test.arpa')
# print(model.score('this is a sentence .', bos = True, eos = True))
# #pip install https://github.com/kpu/kenlm/archive/master.zip
