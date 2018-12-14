import torch
import torch.nn as nn
import numpy as np
import pyflakes
from utils.data_utils import *
from utils.opt_utils import onehot
import warnings
warnings.filterwarnings('ignore')


def make_embeddings(config, vocab):
    padding_idx = vocab.stoi[PAD_WORD]
    word_num = vocab.__len__()
    embeddings = nn.Embedding(word_num, config.word_vec_size, padding_idx=padding_idx)
    vectors = torchtext.vocab.GloVe(name='6B', dim=config.word_vec_size)
    vocab.load_vectors(vectors)
    embeddings.weight.data.copy_(vocab.vectors)
    return embeddings


def _built_rnn(input_size, hidden_dim, dropout=0, type='lstm'):
    if type == 'lstm':
        rnn = nn.LSTM(input_size, hidden_dim, dropout=dropout, bidirectional=False)
    elif type == 'bilstm':
        rnn = nn.LSTM(input_size, hidden_dim//2, dropout=dropout, bidirectional=True)
    else:
        raise ValueError('wrong rnn type:', type)
    return rnn

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attn_type = config.attn_type
        assert (self.attn_type in ['dot', 'general', 'mlp']), ('wrong attention type in config')
        self.input_dim = config.decoder_hidden_dim
        self.context_dim = config.encoder_hidden_dim
        self.attn_hidden_dim = config.attn_hidden_dim
        if self.attn_hidden_dim is not None:
            self.transform = nn.Sequential(
                nn.Linear(self.input_dim, self.attn_hidden_dim),
            )
        matrix_dim = self.attn_hidden_dim if self.attn_hidden_dim is not None else self.input_dim
        if self.attn_type == 'general':
            self.matrix_in = nn.Linear(matrix_dim, matrix_dim, bias=False)
        elif self.attn_type == 'mlp':
            self.input_matrix = nn.Linear(self.input_dim, matrix_dim, bias=False)
            self.ctx_matrix = nn.Linear(self.context_dim, matrix_dim)
            self.v = nn.Linear(matrix_dim, 1, bias=False)
        out_bias = self.attn_type == 'mlp'
        self.transform_linear = nn.Linear(config.decoder_hidden_dim+config.encoder_hidden_dim, config.decoder_hidden_dim, bias=out_bias)
    def get_scores(self, inputs, context):
        if self.attn_type == 'dot':
            if self.attn_hidden_dim is not None:
                inputs = self.transform(inputs)
                context = self.transform(context)
            inputs = inputs.unsqueeze(1)
            scores = torch.bmm(inputs, context.transpose(1, 2)).squeeze(1)

        elif self.attn_type == 'general':
            if self.attn_hidden_dim is not None:
                inputs = self.transform(inputs)
                context = self.transform(context)
            scores = torch.bmm(self.matrix_in(inputs).unsqueeze(1), context.transpose(1, 2)).squeeze(1)
        elif self.attn_type == 'mlp':
            if self.attn_hidden_dim is None:
                self.attn_hidden_dim = inputs.size()[-1]
            batch, source_step, _ = context.size()
            scores = self.v(torch.tanh(self.input_matrix(inputs).unsqueeze(1).unsqueeze(1).expand([batch, 1, source_step, -1]) +
                             self.ctx_matrix(context).unsqueeze(1))).squeeze(-1)
        else:
            raise ValueError('wrong attention type: ', self.attn_type)
        batch, source_step, _ = context.size()
        scores = scores.reshape(batch, -1)
        batch_, source_step_ = scores.size()
        assert batch == batch_ and source_step == source_step_
        return scores

    def forward(self, inputs, context, context_mask=None, ignore_small=0):
        assert inputs.dim() == 2 and context.dim() == 3
        #   context_mask batch*sourceL
        #   context batch*sourceL*sourceH
        #   inputs  batch*targeth
        context = context.transpose(0, 1)
        batch, encoder_step, encoder_hidden_dim = context.size()
        batch_, decoder_hidden_dim = inputs.size()
        assert batch == batch_
        scores = self.get_scores(inputs, context)
        batch1, encoder_step1 = scores.size()
        batch2, encoder_step2 = context_mask.size()

        assert context_mask is not None and batch1 == batch2 and encoder_step1 == encoder_step2, (batch1, batch2, encoder_step1, encoder_step2)
        scores.data.masked_fill_(context_mask, -float('inf'))
        scores = nn.functional.softmax(scores)
        if ignore_small > 0:
            scores = nn.functional.threshold(scores, ignore_small, 0)
        content = torch.bmm(scores.unsqueeze(1), context).squeeze(1)
        concat_c = torch.cat([inputs, content], 1)
        final_h = self.transform_linear(concat_c)
        if self.attn_type in ['dot', 'general']:
            final_h = nn.Tanh(final_h)
        return final_h, scores, content

class Encoder(nn.Module):
    def __init__(self, config, embeddings):
        super(Encoder, self).__init__()
        self.embeddings = embeddings
        self.rnn = _built_rnn(config.word_vec_size, config.encoder_hidden_dim, config.dropout, config.encoder_type)

    def forward(self, inputs, length, init_hidden=None):
        emb = self.embeddings(inputs)
        assert length is not None
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, length)
        encoder_outputs, last_hidden_state = self.rnn(packed_emb, init_hidden)
        encoder_outputs = nn.utils.rnn.pad_packed_sequence(encoder_outputs)[0]
        return encoder_outputs, last_hidden_state
class GeneralDecoder(nn.Module):
    def __init__(self, config, embeddings):
        super(GeneralDecoder, self).__init__()
        self.embeddings = embeddings
        self.attention = Attention(config)
        self.rnn = _built_rnn(config.word_vec_size, config.decoder_hidden_dim, config.dropout)
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.decoder_hidden_dim = config.decoder_hidden_dim

    def apply_mask(self, q):
        # content_mask is size of (batch * slen)
        self.content_mask = q.transpose(0, 1).data.eq(PAD)

    def forward(self, inputs, context, length, init_hidden=None):
        """
        :param inputs:
        :param context:
        :param length:
        :param init_hidden:
        :return: raw decoder hidden state(list), attn scores, content list
        """
        emb = self.embeddings(inputs)
        code_step, batch, emb_dim = emb.size()
        if init_hidden is None:
            init_hidden = [torch.cuda.FloatTensor().new_zeros(1, batch, self.decoder_hidden_dim),
                           torch.cuda.FloatTensor().new_zeros(1, batch, self.decoder_hidden_dim)]
        else:
            init_hidden = [init_hidden[0].reshape(1, batch, -1),
                           init_hidden[1].reshape(1, batch, -1)]
        decoder_hidden = init_hidden
        raw_decoder_hidden, out_decoder_hidden, scores, content = [], [], [], []
        # _, __, init_content = self.attention.forward(decoder_hidden[0].squeeze(), context, self.content_mask)
        for i in range(code_step):
            if i == 0:
                inputs = torch.cat([emb[i], torch.cuda.FloatTensor().new_zeros(batch, self.encoder_hidden_dim)], 1).reshape(1, batch, -1)
            else:
                inputs = torch.cat([emb[i], tmp_content], 1).reshape(1, batch, -1)
            decoder_hidden, _ = self.rnn(inputs, decoder_hidden)
            tmp_out_decoder_hidden, tmp_scores, tmp_content = self.attention.forward(decoder_hidden.squeeze(), context, self.content_mask)
            raw_decoder_hidden.append(decoder_hidden.reshape(batch, -1))
            out_decoder_hidden.append(tmp_out_decoder_hidden)
            scores.append(tmp_scores)
            content.append(tmp_content)
            if tmp_out_decoder_hidden.dim()==2:
                tmp_out_decoder_hidden = tmp_out_decoder_hidden.reshape(1, batch, -1)
            decoder_hidden = _
        raw_decoder_hidden = torch.stack(raw_decoder_hidden)
        out_decoder_hidden = torch.stack(out_decoder_hidden)
        scores = torch.stack(scores)
        content = torch.stack(content)

        code_step_, batch_, hidden_dim = raw_decoder_hidden.size()
        code_step__, batch__, encoder_dim_ = content.size()
        assert code_step_ == code_step and code_step__ == code_step and batch==batch_ and batch==batch__ and\
               hidden_dim==self.decoder_hidden_dim and encoder_dim_==self.encoder_hidden_dim
        return raw_decoder_hidden, out_decoder_hidden, scores, content

class AttnDecoder(nn.Module):
    def __init__(self, config, embeddings):
        super(AttnDecoder, self).__init__()
        self.embeddings = embeddings
        self.attention = Attention(config)
        self.rnn = _built_rnn(config.word_vec_size + config.decoder_hidden_dim, config.decoder_hidden_dim, config.dropout)
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.decoder_hidden_dim = config.decoder_hidden_dim

    def apply_mask(self, q):
        # content_mask is size of (batch * slen)
        self.content_mask = q.transpose(0, 1).data.eq(PAD)

    def forward(self, inputs, context, length, init_hidden=None, init_content=None):
        """
        :param inputs:
        :param context:
        :param length:
        :param init_hidden:
        :return: raw decoder hidden state(list), attn scores, content list
        """

        emb = self.embeddings(inputs)
        code_step, batch, emb_dim = emb.size()
        if init_hidden is None:
            init_hidden = [torch.cuda.FloatTensor().new_zeros(1, batch, self.decoder_hidden_dim),
                           torch.cuda.FloatTensor().new_zeros(1, batch, self.decoder_hidden_dim)]
        else:
            init_hidden = [init_hidden[0].reshape(1, batch, -1),
                           init_hidden[1].reshape(1, batch, -1)]
        decoder_hidden = init_hidden
        raw_decoder_hidden, out_decoder_hidden, scores, content = [], [], [], []
        # _, __, init_content = self.attention.forward(decoder_hidden[0].squeeze(), context, self.content_mask)
        for i in range(code_step):
            if i == 0:
                if init_content is None:
                    inputs = torch.cat([emb[i], torch.cuda.FloatTensor().new_zeros(batch, self.encoder_hidden_dim)], 1).reshape(1, batch, -1)
                else:
                    inputs = torch.cat([emb[i], init_content[0]], 1).reshape(1, batch, -1)
            else:
                inputs = torch.cat([emb[i], tmp_content], 1).reshape(1, batch, -1)

            decoder_hidden, _ = self.rnn(inputs, decoder_hidden)
            tmp_out_decoder_hidden, tmp_scores, tmp_content = self.attention.forward(decoder_hidden.squeeze(), context, self.content_mask)
            raw_decoder_hidden.append(decoder_hidden.reshape(batch, -1))
            out_decoder_hidden.append(tmp_out_decoder_hidden)
            scores.append(tmp_scores)
            content.append(tmp_content)
            if tmp_out_decoder_hidden.dim()==2:
                tmp_out_decoder_hidden = tmp_out_decoder_hidden.reshape(1, batch, -1)
            decoder_hidden = _
        raw_decoder_hidden = torch.stack(raw_decoder_hidden)
        out_decoder_hidden = torch.stack(out_decoder_hidden)
        scores = torch.stack(scores)
        content = torch.stack(content)

        code_step_, batch_, hidden_dim = raw_decoder_hidden.size()
        code_step__, batch__, encoder_dim_ = content.size()
        assert code_step_ == code_step and code_step__ == code_step and batch==batch_ and batch==batch__ and\
               hidden_dim==self.decoder_hidden_dim and encoder_dim_==self.encoder_hidden_dim
        return raw_decoder_hidden, out_decoder_hidden, scores, content, decoder_hidden

class CopyNetwork(nn.Module):
    def __init__(self, dropout, decoder_hidden_dim, context_size, tgt_num, copy_num):
        super(CopyNetwork, self).__init__()
        self.dropout = dropout
        self.transform_linear = nn.Linear(decoder_hidden_dim+context_size, decoder_hidden_dim, bias=True)
        # self.decoder_output_linear = nn.Linear(decoder_hidden_dim, tgt_num)
        self.decoder_output_softmax = nn.Sequential(
            nn.Linear(decoder_hidden_dim, tgt_num),
            nn.LogSoftmax(-1)
        )
        self.copy_linear = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 1),
            nn.Sigmoid()
        )
        self.copy_num = copy_num

    def forward(self, raw_decoder_hidden, out_decoder_hidden, content, scores, copy_to_ext, fields):
        """
        :param raw_decoder_hidden: tlen*batch*decoder_hidden_dim
        :param content: tlen*batch*encoder_hidden_dim
        :param scores: attention scores
        :param copy_to_ext: dict
        :param copy_to_tgt: dict
        :return:
        """
        code_step, batch, decoder_dim = raw_decoder_hidden.size()
        code_step_, batch_, encoder_dim = content.size()
        assert code_step == code_step_ and batch == batch_
        # raw_output_probs = self.decoder_output_linear(out_decoder_hidden)
        raw_output_probs_log = self.decoder_output_softmax(out_decoder_hidden)
        # ========================== calculate copy scores and gen/copy probs =============================
        copy_scores = self.copy_linear(raw_decoder_hidden)

        # gen_output_probs_log = torch.mul(raw_output_probs_log, (1.0-copy_scores).expand_as(raw_output_probs_log))
        gen_output_probs_log = raw_output_probs_log
        def safe_log(v):
            return torch.log(v.clamp(1e-6, 1 - 1e-6))
        scores = torch.mul(scores, copy_scores.expand_as(scores))
        copy_to_ext_onehot = onehot(copy_to_ext, N=self.copy_num, ignore_index=fields['copy_to_ext'].vocab.stoi[UNK_WORD])
        copy_output_probs = torch.bmm(scores.transpose(0, 1), copy_to_ext_onehot.transpose(0, 1)).transpose(0, 1)
        # copy_output_probs_log = torch.mul(safe_log(copy_output_probs), copy_scores.expand_as(copy_output_probs))
        copy_output_probs_log = safe_log(copy_output_probs)
        # copy_output_probs_log = torch.log(copy_output_probs)
        all_output_probs = torch.cat([gen_output_probs_log, copy_output_probs_log], 2)
        return all_output_probs, copy_scores

class LinearClassifer(nn.Module):
    def __init__(self, dropout, decoder_hidden_dim, context_size, tgt_num, copy_num):
        """
        for test and baseline
        :param dropout:
        :param decoder_hidden_dim:
        :param context_size:
        :param tgt_num:
        """
        super(LinearClassifer, self).__init__()
        self.dropout = dropout
        self.decoder_output_linear = nn.Sequential(
            nn.Linear(decoder_hidden_dim, tgt_num),
            nn.LogSoftmax(-1)
        )
    def forward(self, raw_decoder_hidden, out_decoder_hidden, content, scores, copy_to_ext, fields):
        code_step, batch, decoder_dim = raw_decoder_hidden.size()
        code_step_, batch_, encoder_dim = content.size()
        assert code_step == code_step_ and batch == batch_
        output_probs_log = self.decoder_output_linear(out_decoder_hidden)
        return output_probs_log, None

class Seq2seqCopyModel(nn.Module):
    def __init__(self, config, encoder, decoder, classifer, fields):
        super(Seq2seqCopyModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifer = classifer
        self.fields = fields

    def forward(self, inputs, inputs_len, tgt_inputs, copy_to_ext, fields):
        slen, batch_size = inputs.size()
        encoder_outputs, hidden_state = self.encoder(inputs, inputs_len)
        raw_decoder_hidden, out_decoder_hidden, scores, content = self.decoder(tgt_inputs, encoder_outputs, inputs_len)
        all_output_probs, copy_scores = self.classifer(raw_decoder_hidden, out_decoder_hidden, content, scores, copy_to_ext, fields)
        # return torch.argmax(all_output_probs, dim=2)
        return all_output_probs, copy_scores

def make_encoder(config, query_embeddings):
    encoder = Encoder(config, query_embeddings)
    return encoder
def make_decoder(config, tgt_embeddings):
    decoder = AttnDecoder(config, tgt_embeddings)
    return decoder
def make_linear_classifer(dropout, decoder_h, encoder_h, tgt_n, copy_n):
    classifer = LinearClassifer(dropout, decoder_h, encoder_h, tgt_n, copy_n)
    return classifer
class Seq2seqModel(nn.Module):
    def __init__(self, config, encoder, decoder, classifer, fields):
        super(Seq2seqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifer = classifer

    def forward(self, inputs, inputs_len, tgt_inputs, copy_to_ext, fields):
        slen, batch_size = inputs.size()
        encoder_outputs, encoder_hidden_state = self.encoder(inputs, inputs_len)
        init_decoder_hidden_state = [torch.cat([encoder_hidden_state[0][0], encoder_hidden_state[0][1]], 1),
                                     torch.cat([encoder_hidden_state[1][0], encoder_hidden_state[1][1]], 1)]
        raw_decoder_hidden, out_decoder_hidden, scores, content, _ = self.decoder(tgt_inputs, encoder_outputs, inputs_len, init_decoder_hidden_state)
        all_output_probs, copy_scores = self.classifer(raw_decoder_hidden, out_decoder_hidden, content, scores, copy_to_ext, fields)
        # return torch.argmax(all_output_probs, dim=2)
        return all_output_probs, copy_scores, scores

def built_model(config, fields):
    query_embeddings = make_embeddings(config, fields['src'].vocab)
    tgt_embeddings = make_embeddings(config, fields['tgt'].vocab)
    encoder = Encoder(config, query_embeddings)
    decoder = AttnDecoder(config, tgt_embeddings)
    if config.classifer == 'linear':
        classifer = LinearClassifer(config.dropout, config.decoder_hidden_dim, config.encoder_hidden_dim,
                        fields['tgt'].vocab.__len__(), fields['copy_to_ext'].vocab.__len__())
    elif config.classifer == 'copy':
        classifer = CopyNetwork(config.dropout, config.decoder_hidden_dim, config.encoder_hidden_dim,
                        fields['tgt'].vocab.__len__(), fields['copy_to_ext'].vocab.__len__())
    else:
        raise ValueError('wrong classifer type')
    model = Seq2seqModel(config, encoder, decoder, classifer, fields)
    model = model.cuda()
    return model