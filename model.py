import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl


###########################################################
# This seq2seq with attention model is copied from
# https://teddykoker.com/2020/02/nlp-from-scratch-annotated-attention/
# Note:
# - We don't use batch_first here, so typically a tensor containing
#   input data is of shape (seq_length, batch_size, hidden_dim).
#   batch size is the 2nd dim.
# - We use pytorch_lightning for simpler training loop.
#   The configure_optimizers function and training_step function
#   in class Seq2Seq are hooks for pl.Trainer in main.py.
###########################################################
class Encoder(pl.LightningModule):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout if n_layers > 1 else 0)

    def forward(self, x):
        embedding = self.drop(self.embed(x))
        output, hidden = self.rnn(embedding)
        return self.drop(output), hidden


class Attention(pl.LightningModule):
    def __init__(self, hidden_size, score='general'):
        super().__init__()
        self.score_fn = score
        if score == 'general':
            self.w = nn.Linear(hidden_size, hidden_size)
        elif score == 'concat':
            self.w = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        elif score != 'dot':
            raise ValueError("score should be 'dot', 'general' or 'concat', '{}' is not supported.".format(score))

    def score(self, decoder_out, encoder_out):
        if self.score_fn == 'dot':
            return torch.sum(decoder_out * encoder_out, dim=2)
        if self.score_fn == 'general':
            return torch.sum(decoder_out * self.w(encoder_out), dim=2)
        if self.score_fn == 'concat':
            decoder_out = decoder_out.repeat(encoder_out.size(0), 1, 1)
            concat = torch.cat((decoder_out, encoder_out))
            return torch.sum(self.v * self.w(concat), dim=2)

    def forward(self, decoder_out, encoder_out):
        score = self.score(decoder_out, encoder_out)
        score = F.softmax(score, dim=0)
        context = torch.bmm(score.transpose(1, 0).unsqueeze(1), encoder_out.transpose(1, 0)).transpose(1, 0)
        return score, context


class Decoder(pl.LightningModule):
    def __init__(self, output_size, embedding_size, hidden_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout if n_layers > 1 else 0)
        self.attn = Attention(hidden_size)
        self.wc = nn.Linear(hidden_size * 2, hidden_size)
        self.ws = nn.Linear(hidden_size, output_size)

    def forward(self, target, hidden, encoder_out):
        target = target.unsqueeze(0)
        embedding = self.drop(self.embed(target))
        decoder_out, hidden = self.rnn(embedding, hidden)
        attn, context = self.attn(decoder_out, encoder_out)

        attn_hidden = self.wc(torch.cat((decoder_out, context), dim=2)).tanh()
        out = self.ws(attn_hidden.squeeze(0))

        return out, hidden, attn


class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(encoder_vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(decoder_vocab_size, embedding_dim, hidden_dim)
        # self.out_length = out_length

    def forward(self, inputs, target):
        # length = self.out_length if self.out_length > 0 else target.size(0)
        out = torch.zeros(target.size(0), target.size(1), self.decoder.output_size).to(inputs.device)
        encoder_out, hidden = self.encoder(inputs)

        x = target[0]
        for t in range(target.size(0)):
            out[t], hidden, _ = self.decoder(x, hidden, encoder_out)
            x = out[t].argmax(1)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, y)
        loss = F.cross_entropy(output.view(-1, output.size(2)), y.view(-1))
        self.log('Train loss', loss)
        return loss

    # def predict(self, x):
    #     output = self(x)
    #     return F.softmax(output, dim=2)
    @torch.no_grad()
    def predict(self, x, predict_length, start=0):
        self.eval()
        y = torch.tensor([start]).long().to(x.device)
        trgs, prob, attention = [], [], []
        encoder_out, hidden = self.encoder(x)

        for t in range(predict_length):
            y, hidden, atten = self.decoder(y, hidden, encoder_out)
            prob.append(F.softmax(y, dim=1))
            y = y.argmax(1)
            trgs.append(y)
            attention.append(atten.T)

        attention = torch.cat(attention).cpu().numpy()
        return trgs, prob, attention
