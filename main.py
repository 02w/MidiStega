import os
import torch
from torch.utils.data import DataLoader
from model import Seq2Seq
from utils import collate_fn, DataParser, NoteDataset
import pytorch_lightning as pl

if __name__ == '__main__':

    # get all filenames in 'data' folder
    dirs = list(os.walk('data'))
    files = []
    for i, author in enumerate(dirs[0][1]):
        for f in dirs[i + 1][2]:
            files.append('{}/{}'.format(author, f))

    # keep_author = True will add author name to the beginning
    data = DataParser(path='data/', file_list=files, keep_author=True)
    # data.note2id maps word to index, e.g. data.note2id['mozart'] = 0
    # data.id2note maps index to word, e.g. data.id2note[0] = 'mozart'

    inputs_data, target_data = data.get_inputs_and_targets(length=10, overlap=0.2)

    train_loader = DataLoader(
        dataset=NoteDataset(inputs_data, target_data),
        batch_size=16,
        collate_fn=collate_fn
    )

    model = Seq2Seq(len(data.note2id), len(data.note2id), 100, 128)

    # to use cuda: trainer = pl.Trainer(max_epochs=?, gpus=1)
    trainer = pl.Trainer(max_epochs=20)

    trainer.fit(model, train_loader)

    # shape of x: (sequence_length, batch_size=1)
    pred, prob, _ = model.predict(
        x=torch.tensor(data.note2id['mozart']).unsqueeze(1),
        y=data.note2id['mozart'],
        predict_length=1
    )

    # prob: [tensor_1, tensor_2, ..., tensor_predict_length]
    # if predict_length = 1, prob = [tensor_1]
    # shape of tensor_i: (1, vocab_size)
    # tensor_i contains probability of each word
