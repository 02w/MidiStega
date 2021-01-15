import os
import torch
from model import Seq2Seq
from utils import DataParser, bin2str, str2bin, group
from convert import midi_to_txt, txt_to_midi, MELODY_NOTE_OFF
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

# get all filenames in 'data' folder
# dirs = list(os.walk('data'))
# files = []
# for i, author in enumerate(dirs[0][1]):
#     for f in dirs[i + 1][2]:
#         files.append('{}/{}'.format(author, f))
#
# keep_author = True will add author name to the beginning
# data = DataParser(path='data/', file_list=files, keep_author=True)

data = DataParser.load('data.joblib')
# load model from .ckpt file
model = Seq2Seq.load_from_checkpoint(
    'versions/version_1/epoch=49.ckpt',
    encoder_vocab_size=len(data.note2id),
    decoder_vocab_size=len(data.note2id),
    embedding_dim=128,
    hidden_dim=256,
    n_layers=2
)


def plot_attn(attn_data):
    # sns.set()
    # sns.heatmap(attn_data)

    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.matshow(attn_data)
    plt.show()
    plt.savefig('attn.png')


def hide(message, window, seq, start_note, output_dir):
    name = os.path.basename(message)

    with open(message, 'rb') as f:
        src = f.read()

    str_list, groups = bin2str(src, window=window)

    # shape of x: (sequence_length, batch_size=1)
    pred, prob, _ = model.predict(
        # prepare a sequence as input here
        x=torch.tensor(seq).unsqueeze(1),
        # choose a start note
        start=start_note,
        predict_length=len(groups) + 1
    )

    # prob: [tensor_1, tensor_2, ..., tensor_predict_length]
    # if predict_length = 1, prob = [tensor_1]
    # tensor_i contains probability of each word

    # print(prob[0])

    # use torch.topk to find the greatest k elements
    # torch.topk(input, k, dim=None,largest=True, sorted=None, out=None0)
    # -> (Tensor, LongTensor)
    result = '52 '
    for index, note in zip(groups, prob[1:]):
        _, candidates = torch.topk(note, 2 ** window, dim=0)
        candidates = candidates.cpu().numpy()
        result += data.id2note[candidates[int(index, 2)]] + ' '

    with open('versions/tmp/hide/' + os.path.splitext(name)[0] + '.txt', 'w', encoding='utf-8') as f:
        f.write(result + str(MELODY_NOTE_OFF))
    txt_to_midi(path='versions/tmp/hide/' + os.path.splitext(name)[0] + '.txt', output_dir=output_dir)


def extract(cover, window, seq, start_note, output_dir):
    name = os.path.basename(cover)

    midi_to_txt(path=cover, output_dir='versions/tmp/extract')
    with open('versions/tmp/extract/' + name.replace('midi', 'txt'), 'r', encoding='utf8') as f:
        notes = f.read().strip().split()

    pred, prob, _ = model.predict(
        # prepare a sequence as input here
        x=torch.tensor(seq).unsqueeze(1),
        # choose a start note
        start=start_note,
        predict_length=len(notes) - 1
    )

    str_list = []
    for note, p in zip(notes[1:], prob[1:]):
        _, candidates = torch.topk(p, 2 ** window, dim=0)
        candidates = candidates.cpu().numpy()
        index = data.note2id[note]
        m = np.where(candidates == index)[0][0]
        str_list.append(np.binary_repr(m, width=window))

    tmp = ''.join(str_list)
    for i in range(len(tmp) - 1, 0, -1):
        if tmp[i] != '0':
            break
    groups = group(tmp[: i], n=8)
    str2bin(groups, output_dir + '/' + os.path.splitext(name)[0] + '.out')


def compose(seq, length, random_choose=True, window=5):
    pred, prob, attn = model.predict(
        # prepare a sequence as input here
        x=torch.tensor(seq).unsqueeze(1),
        # choose a start note
        start=seq[0],
        predict_length=length
    )
    result = ''
    for p in prob[1:]:
        _, candidates = torch.topk(p, window, dim=0)
        candidates = candidates.cpu().numpy()
        if random_choose:
            result += data.id2note[candidates[random.randint(0, window - 1)]] + ' '
        else:
            result += data.id2note[candidates[0]] + ' '

    with open('versions/tmp/compose.txt', 'w', encoding='utf8') as f:
        f.write(result + str(MELODY_NOTE_OFF))

    txt_to_midi(path='versions/tmp/compose.txt', output_dir='versions/midi')
    plot_attn(attn.cpu().numpy())


if __name__ == '__main__':
    hide(
        message='versions/msg/bin',
        window=5,
        seq=[data.note2id['Franz Schubert']] + [data.note2id[i] for i in data.melodies[6]][: 128],
        start_note=data.note2id['Franz Schubert'],
        output_dir='versions/midi'
    )

    extract(
        cover='versions/midi/bin.midi',
        window=5,
        seq=[data.note2id['Franz Schubert']] + [data.note2id[i] for i in data.melodies[6]][: 128],
        start_note=data.note2id['Franz Schubert'],
        output_dir='versions/msg'
    )
    
    compose(
        seq=[data.note2id['Franz Schubert']] + [data.note2id[i] for i in data.melodies[99]][: 25],
        length=25,
        random_choose=True,
        window=8
    )
