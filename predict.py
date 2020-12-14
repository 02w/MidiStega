import os
import torch
from model import Seq2Seq
from utils import DataParser

if __name__ == '__main__':
    # get all filenames in 'data' folder
    dirs = list(os.walk('data'))
    files = []
    for i, author in enumerate(dirs[0][1]):
        for f in dirs[i + 1][2]:
            files.append('{}/{}'.format(author, f))

    # keep_author = True will add author name to the beginning
    data = DataParser(path='data/', file_list=files[:10], keep_author=True)

    # load model from .ckpt file
    model = Seq2Seq.load_from_checkpoint(
        'lightning_logs/version_1/checkpoints/epoch=0.ckpt',
        encoder_vocab_size=len(data.note2id),
        decoder_vocab_size=len(data.note2id),
        embedding_dim=100,
        hidden_dim=128
    )
    # shape of x: (sequence_length, batch_size=1)
    pred, prob, _ = model.predict(
        x=torch.tensor([data.note2id['mozart']]).unsqueeze(1),
        start=data.note2id['mozart'],
        predict_length=1
    )

    # prob: [array_1, array_2, ..., array_predict_length]
    # if predict_length = 1, prob = [array_1]
    # array_i contains probability of each word

    print(prob[0])
