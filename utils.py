import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from nltk import ngrams


def collate_fn(data):
    inputs, targets = map(list, zip(*data))
    inputs = pad_sequence(inputs)
    targets = pad_sequence(targets)
    # if torch.cuda.is_available():
    #     inputs = inputs.cuda()
    return inputs, targets


class NoteDataset(Dataset):
    def __init__(self, inputs, targets, train=True):
        self.inputs = inputs
        self.targets = targets
        self.train = train

    def __getitem__(self, item):
        return torch.LongTensor(self.inputs[item]), torch.LongTensor(self.targets[item])

    def __len__(self):
        return len(self.inputs)


class DataParser(object):
    def __init__(self, path, file_list, keep_author=True):
        super().__init__()
        self.path = path
        self.files = file_list
        self.keep_author = keep_author
        self.melodies = []
        self.note2id = {}
        self.id2note = {}
        self.__read_notewise_files()
        self.__parse_vocab()

    def __read_notewise_files(self):
        for file in self.files:
            if self.keep_author:
                s = file.split('/')
                if len(s) != 2:
                    raise ValueError("If keep_author is True, names in file_list should be of format 'author/example.txt'.")
                author, _ = s
            with open(self.path + '/' + file, encoding='utf-8') as f:
                for line in f:
                    if self.keep_author:
                        self.melodies.append([author] + line.strip().split())
                    else:
                        self.melodies.append(line.strip().split())

    def __parse_vocab(self):
        for line in self.melodies:
            for note in line:
                if self.note2id.get(note) is None:
                    self.note2id[note] = len(self.note2id)

        self.id2note = {v: k for k, v in self.note2id.items()}

    def get_ngrams(self, ngram):
        if self.keep_author:
            ret = []
            for i in self.melodies:
                ngs = list(ngrams(i[1:], ngram))
                ret.append([[i[0]] + list(j) for j in ngs])
                return ret
        else:
            return [list(ngrams(i, ngram)) for i in self.melodies]

    def get_inputs_and_targets(self, length, overlap=0.5):
        # data = []
        inputs_data = []
        target_data = []
        gap = int(length * (1 - overlap))
        for ng in self.get_ngrams(length):
            for i in range(0, len(ng) - gap, gap):
                inputs_data.append([self.note2id[n] for n in ng[i]])
                target_data.append([self.note2id[n] for n in ng[i + gap]])
                # data.append((ng[i], ng[i + gap]))
        # for d in data:
        #     train_data.append([note2id[i] for i in d[0]])
        #     target_data.append([note2id[i] for i in d[1]])
        return inputs_data, target_data
