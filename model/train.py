import os
import torch
from torch.utils.data import DataLoader
from seq2seq import Seq2Seq
from utils import collate_fn, DataParser, NoteDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':

    # keep_author = True will add author name to the beginning
    data = DataParser(path='data/', keep_author=True)
    # data.note2id maps word to index, e.g. data.note2id['Beethoven'] = 0
    # data.id2note maps index to word, e.g. data.id2note[0] = 'Beethoven'
    #data.save('data.pkl')

    inputs_data, target_data = data.get_inputs_and_targets(length=256, overlap=0.3)

    train_cnt = int(len(inputs_data) * 0.8)
    train_loader = DataLoader(
        dataset=NoteDataset(inputs_data[: train_cnt], target_data[: train_cnt]),
        batch_size=32,
        collate_fn=collate_fn
    )

    validation_loader = DataLoader(
        dataset=NoteDataset(inputs_data[train_cnt:], target_data[train_cnt:]),
        batch_size=32,
        collate_fn=collate_fn
    )

    model = Seq2Seq(
        encoder_vocab_size=len(data.note2id),
        decoder_vocab_size=len(data.note2id),
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2
    )

    checkpoint_callback = ModelCheckpoint(dirpath='versions/checkpoints', period=2, save_top_k=-1)

    trainer = pl.Trainer(max_epochs=100, gpus=1, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, validation_loader)

    # tensorboard --logdir ./lightning_logs/version_0
