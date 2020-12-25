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
    # data.note2id maps word to index, e.g. data.note2id['Franz Schubert'] = 0
    # data.id2note maps index to word, e.g. data.id2note[0] = 'Franz Schubert'
    data.save('data.joblib')

    inputs_data, target_data = data.get_inputs_and_targets(length=64, overlap=0.2)

    train_loader = DataLoader(
        dataset=NoteDataset(inputs_data, target_data),
        batch_size=16,
        collate_fn=collate_fn
    )

    model = Seq2Seq(
        encoder_vocab_size=len(data.note2id),
        decoder_vocab_size=len(data.note2id),
        embedding_dim=800,
        hidden_dim=128
    )

    # to use cuda: trainer = pl.Trainer(max_epochs=?, gpus=1)
    trainer = pl.Trainer(max_epochs=1)

    trainer.fit(model, train_loader)
