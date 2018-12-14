import torch
import torch.nn as nn
import torchtext
import os
from utils.data_utils import *
from evaluate import count_accuracy
from model import LinearClassifer, CopyNetwork

class SupervisedTrainer:
    def __init__(self, model, fields, train_iter, val_iter, config):
        self.model = model
        self.fields = fields
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.config = config
        # self.model.train()
        if config.criterion == 'cross_entropy':
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=PAD)
        else:
            raise ValueError('wrong criterion type')
        self.params = [i for i in self.model.parameters() if i.requires_grad]
        if config.optimizer == 'Adam':
            self.optimizor = torch.optim.Adam(self.params, lr=config.learning_rate,
                                              betas=[0.9, 0.98], eps=1e-9)

    def cal_loss(self, logit, label, mask=None):
        loss_sum = []
        for i, (pred, true) in enumerate(zip(logit, label)):
            # if mask is not None:
            #     true = torch.mul(true, mask.long())
            loss = self.criterion(pred.cuda(), true.cuda())
            loss_sum.append(loss)
        return sum(loss_sum)
    def forward(self, i, batch, fields):
        # ================== forward propagation =========================================
        query, query_len = batch.src
        if query.size()[0] >100:
            aa = 1
        tgt, tgt_len = batch.tgt
        if tgt.size()[0]>100:
            aa = 1
        copy_to_ext = batch.copy_to_ext
        self.model.decoder.apply_mask(query)
        tgt_output, copy_scores, attn_scores = self.model(query, query_len, tgt[:-1], copy_to_ext, fields)
        if isinstance(self.model.classifer, CopyNetwork):
            assert tgt_output.size()[2] == fields['tgt'].vocab.__len__() + fields['copy_to_ext'].vocab.__len__()
            # ================== calculate loss ==============================================
            tgt_copy_mask = batch.tgt_copy_ext.ne(fields['tgt_copy_ext'].vocab.stoi[UNK_WORD]).long()[1:]
            tgt_gen_mask = batch.tgt_copy_ext.eq(fields['tgt_copy_ext'].vocab.stoi[UNK_WORD]).long()[1:]
            tgt_label = torch.mul(tgt_copy_mask, batch.tgt_copy_ext[1:] + fields['tgt'].vocab.__len__()) +\
            torch.mul(tgt_gen_mask, tgt[1:])
        elif isinstance(self.model.classifer, LinearClassifer):
            assert tgt_output.size()[2] == fields['tgt'].vocab.__len__()
            tgt_label = tgt[1:]
        loss = self.cal_loss(tgt_output, tgt_label.cuda(), tgt_label.eq(PAD).cuda())
        (correct_m_and_count) = count_accuracy(tgt_output, tgt_label.cuda(), mask=tgt_label.eq(PAD).cuda(), row=True)
        return loss, correct_m_and_count

    def train(self, epoch):
        correct_count = 0
        count_all = 0
        loss_all = 0
        i=-1
        self.model.train()
        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()
            loss, correct_m_and_count = self.forward(epoch, batch, self.fields)

            loss.backward()
            self.optimizor.step()
            loss_all += loss.data
            correct_count += correct_m_and_count[0].sum()
            count_all += correct_m_and_count[1]
            if i>0 and i%10 == 0:
                print('Batch %d, loss_avg: %.6f, train acc is: %.6f' % (i, loss_all/i, correct_count/count_all))
        correct_rate = correct_count/ count_all
        return loss_all/i, correct_rate

    def validate(self, epoch):
        self.model.eval()
        loss_all = 0
        count_all = 0
        correct_count = 0
        i = -1
        for i, batch in enumerate(self.val_iter):
            loss, correct_m_and_count = self.forward(epoch, batch, self.fields)
            loss_all += loss.data
            correct_count += correct_m_and_count[0].sum()
            count_all += correct_m_and_count[1]
        correct_rate = correct_count/ count_all
        return loss_all/i, correct_rate

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    def save_checkpoint(self, config, epoch, fields):
        model_state_dict = self.model.state_dict()

        check_point = {
            'model': model_state_dict,
            'vocab': self.save_vocab(fields),
            'config': config.__dict__,
            'epoch': epoch,
            'optim': self.optimizor,
        }
        torch.save(check_point, os.path.join(config.save_path, 'model_%s.pt' % epoch))