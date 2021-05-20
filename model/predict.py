import os

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from .convert import midi2seq, seq2midi, MELODY_NOTE_OFF
from .seq2seq import Seq2Seq
from .utils import DataParser, bin2str, str2bin, group

# data = DataParser('model/data')
# data.save('data.pkl')
data = DataParser.load('model/data.pkl')
model = Seq2Seq(
    encoder_vocab_size=len(data.note2id),
    decoder_vocab_size=len(data.note2id),
    embedding_dim=128,
    hidden_dim=256,
    n_layers=2
)
model.load_state_dict(torch.load('model/model.pkl'))


def choose(p, window, group):
    _, candidates = torch.topk(p, 2 ** window, dim=0)
    return candidates[int(group, 2)]


def hide(message, window, seq, output_dir):
    str_list, groups = bin2str(message, window=window)

    # shape of x: (sequence_length, batch_size=1)
    pred, prob, _ = model.predict(
        # prepare a sequence as input here
        x=torch.tensor(seq).unsqueeze(1),
        # choose a start note
        start=seq[0],
        predict_length=len(groups),
        topk=window,
        # how to choose a note from the probability distribution
        criteria=choose,
        groups=groups
    )

    result = [52] + [data.id2note[p.item()] for p in pred] + [MELODY_NOTE_OFF]
    joblib.dump(result, 'tmp/hide/tmp.pkl')

    return seq2midi(path='tmp/hide/tmp.pkl', out_dir=output_dir)


def extract(cover, window, seq, output_dir):
    name = os.path.basename(cover)

    midi2seq(path=cover, out_dir='tmp/extract')
    notes = joblib.load('tmp/extract/' + name.replace('midi', 'pkl'))

    start_note = seq[0]
    predict_length = len(notes) - 1  # - 1: skip the beginning '52'
    str_list = []

    model.eval()
    x = torch.tensor(seq).unsqueeze(1)
    y = torch.tensor([start_note]).long().to(x.device)

    encoder_out, hidden = model.encoder(x)

    y, hidden, atten = model.decoder(y, hidden, encoder_out)
    y = y.argmax(1)

    for t in range(predict_length - 1):  # ignore the added MELODY_NOTE_OFF
        y, hidden, atten = model.decoder(y, hidden, encoder_out)
        p = F.softmax(y, dim=1).squeeze(0)

        _, candidates = torch.topk(p, 2 ** window, dim=0)
        candidates = candidates.cpu().numpy()
        y = data.note2id[notes[t + 1]]
        try:
            m = np.where(candidates == y)[0][0]
            str_list.append(np.binary_repr(m, width=window))
        except IndexError:
            print(f'Error at step {t}: only part of the message can be extracted!')

        y = torch.tensor([y])

    tmp = ''.join(str_list)
    for i in range(len(tmp) - 1, 0, -1):
        if tmp[i] != '0':
            break
    groups = group(tmp[: i], n=8)
    save_path = output_dir + '/' + os.path.splitext(name)[0] + '.out'
    str2bin(groups, save_path)
    return save_path


def hide_app(message, seq_index):
    seq = data.melodies[seq_index][: 128]
    seq = [data.note2id[i] for i in seq]
    return hide(message, window=3, seq=seq, output_dir='tmp/hide/')


def extract_app(cover, seq_index):
    seq = data.melodies[seq_index][: 128]
    seq = [data.note2id[i] for i in seq]
    return extract(cover, window=3, seq=seq, output_dir='tmp/extract')


if __name__ == '__main__':
    hide(
        message='versions/msg/123.docx',
        window=5,
        seq=[data.note2id[i] for i in data.melodies[6]][: 128],
        output_dir='versions/midi'
    )

    extract(
        cover='versions/midi/123.midi',
        window=5,
        seq=[data.note2id[i] for i in data.melodies[6]][: 128],
        output_dir='versions/msg'
    )
