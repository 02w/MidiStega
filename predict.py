import os
import torch
from model import Seq2Seq
from utils import DataParser
import random


if __name__ == '__main__':
    # get all filenames in 'data' folder
    dirs = list(os.walk('data'))
    files = []
    for i, author in enumerate(dirs[0][1]):
        for f in dirs[i + 1][2]:
            files.append('{}/{}'.format(author, f))

    # keep_author = True will add author name to the beginning
    data = DataParser(path='data/', file_list=files, keep_author=True)

    # load model from .ckpt file
    model = Seq2Seq.load_from_checkpoint(
        'versions/version_0/checkpoints/epoch=99.ckpt',
        encoder_vocab_size=len(data.note2id),
        decoder_vocab_size=len(data.note2id),
        embedding_dim=800,
        hidden_dim=128
    )
    # shape of x: (sequence_length, batch_size=1)
    pred, prob, _ = model.predict(
        # prepare a sequence as input here
        x=torch.tensor([data.note2id['Franz Schubert']] + [data.note2id[i] for i in data.melodies[0]]).unsqueeze(1),
        # choose a start note
        start=data.note2id['Franz Schubert'],
        predict_length=1500
    )

    # prob: [tensor_1, tensor_2, ..., tensor_predict_length]
    # if predict_length = 1, prob = [tensor_1]
    # tensor_i contains probability of each word

    # print(prob[0])

    # use torch.topk to find the greatest k elements
    # torch.topk(input, k, dim=None,largest=True, sorted=None, out=None0)
    # -> (Tensor, LongTensor)
    result = ''
    for note in prob:
        _, index = torch.topk(note, 5, dim=0)
        index = index.cpu().numpy()
        result += data.id2note[index[random.randint(0, 4)]] + ' '

    with open('versions/midi/result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
