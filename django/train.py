import torch
import torch.nn as nn
import torchtext
from preprocess import built_dataset_vocab
from model import Seq2seqModel, Encoder, AttnDecoder, LinearClassifer, CopyNetwork
from model import make_embeddings, built_model
from utils.data_utils import OrderedIterator, TableDataset
import config as _config
from SupervisedTrainer import SupervisedTrainer
from ReinforcedTrainer import PolicyGradientTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_fields(config, train, valid, checkpoint):
    fields = TableDataset.load_fields(
        torch.load(os.path.join(config.django_data_path, 'vocab.pt')))
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    if config.train_from_pg>0:
        print('Loading vocab from checkpoint at %s.' % config.train_from_pg)
        fields = TableDataset.load_fields(checkpoint['vocab'])
    elif config.train_from_sv>0:
        print('Loading vocab from checkpoint at %s.' % config.train_from_sv)
        fields = TableDataset.load_fields(checkpoint['vocab'])

    return fields





def train_model_by_svlearning(config, model, fields, train, dev):
    train_iter = OrderedIterator(train, config.batch_size, sort_within_batch=True, repeat=False)
    dev_iter = OrderedIterator(dev, config.batch_size, train=False,
                               repeat=False, sort_within_batch=True)

    sv_trainer = SupervisedTrainer(model, fields, train_iter, dev_iter, config)
    last_dev_loss = None
    epoch = -1
    for epoch in range(config.train_from_pg, config.max_epochs):
        train_loss, train_correct_rate = sv_trainer.train(epoch)

        dev_loss, dev_correct_rate = sv_trainer.validate(epoch)
        if last_dev_loss is None:
            last_dev_loss = dev_loss
        if epoch%1 == 0:
            print('Epoch %d/%d:\n train loss is %.6f, train acc is %.6f\n'
                  'val loss is %.6f, val acc is %.6f\n\n' %
                  (epoch, config.max_epochs, train_loss, train_correct_rate, dev_loss, dev_correct_rate))
        if epoch >= config.save_after_epoch:
            sv_trainer.save_checkpoint(config, epoch, fields)
    if epoch == config.max_epochs-1:
        sv_trainer.save_checkpoint(config, epoch, fields)
    print('Training all done in epoch %d\n'%epoch)

def train_model_by_pglearning(config, model, fields, train, dev):
    train_iter = OrderedIterator(train, config.batch_size, sort_within_batch=True, repeat=False)
    dev_iter = OrderedIterator(dev, config.batch_size, train=False,
                               repeat=False, sort_within_batch=True)

    pg_trainer = PolicyGradientTrainer(model, fields, train_iter, dev_iter, config)
    last_dev_loss = None
    epoch = -1
    for epoch in range(config.train_from_pg, config.max_epochs):
        train_loss, train_correct_rate = pg_trainer.train(epoch)

        dev_loss, dev_correct_rate = pg_trainer.validate(epoch)
        if last_dev_loss is None:
            last_dev_loss = dev_loss
        if epoch%1 == 0:
            print('Epoch %d/%d:\n train loss is %.6f, train acc is %.6f\n'
                  'val loss is %.6f, val acc is %.6f\n\n' %
                  (epoch, config.max_epochs, train_loss, train_correct_rate, dev_loss, dev_correct_rate))
        assert config.save_after_epoch >= config.train_from_pg
        if epoch >= config.save_after_epoch:
            pg_trainer.save_checkpoint(config, epoch, fields)
    if epoch == config.max_epochs-1:
        pg_trainer.save_checkpoint(config, epoch, fields)
    print('Training all done in epoch %d\n' % epoch)
if __name__ == '__main__':
    checkpoint = None
    config = _config.django_param()
    # =========================== policy gradient =========================================

    # ================== read from file ===============================
    # train = torch.load(os.path.join(config.django_data_path, 'train.pt'))
    # dev = torch.load(os.path.join(config.django_data_path, 'dev.pt'))

    # ================== preprocess file ==============================
    train, dev, test, fields = built_dataset_vocab(config, config.train_js_path, config.dev_js_path, config.test_js_path)
    # fields = load_fields(train, dev, checkpoint)
    train.fields = fields
    dev.fields = fields

    print('train examples number is:%d' % train.examples.__len__())
    print('val examples number is:%d' % dev.examples.__len__())
    print('test examples number is:%d' % test.examples.__len__())

    model = built_model(config, fields)
    # train_model_by_svlearning(config, model, fields, train, dev)
    train_model_by_pglearning(config, model, fields, train, dev)