import torch.nn as nn
import torch
import torchtext
import os
from utils.data_utils import *
import config as config_
from preprocess import read_anno_json, built_dataset_vocab
from model import built_model
from evaluate import count_accuracy, count_accuracy_int
class Result:
    def __init__(self, idx, data):
        self.idx = idx
        self.data = data
class Generator:
    def __init__(self, model, checkpoint):


        assert checkpoint is not None
        c = torch.load(checkpoint, map_location=lambda x, y: x)

        self.fields = TableDataset.load_fields(c['vocab'])
        config = c['config']
        extra_config = config_.django_param()
        for key in extra_config.__dict__:
            if key not in config:
                config[key] = extra_config.__dict__[key]
        self.config = config_.django_param()
        for key, value in config.items():
            setattr(self.config, key, value)
        if model is None:
            self.model = built_model(self.config, self.fields)
        else:
            self.model = model
        self.model.load_state_dict(c['model'])
        self.model.eval()

    def run_decoder(self, encoder_outputs, encoder_hidden_state, batch_size, inputs_len, copy_to_ext):
        decoder_hidden_state_tuple = [torch.cat([encoder_hidden_state[0][0], encoder_hidden_state[0][1]], 1),
                                     torch.cat([encoder_hidden_state[1][0], encoder_hidden_state[1][1]], 1)]
        inp = torch.LongTensor(1, batch_size).fill_(BOS).cuda()
        dec_list, dec_topk = [], []
        content, init_content = None, None
        for i in range(self.config.max_dec_step):
            if i>0:
                init_content = content
            inp.masked_fill_(inp.ge(fields['tgt'].vocab.__len__()), UNK)
            raw_decoder_hidden, out_decoder_hidden, scores, content, decoder_hidden_state_tuple\
                = self.model.decoder(inp, encoder_outputs, inputs_len, decoder_hidden_state_tuple, init_content)
            all_prob, copy_scores = self.model.classifer(raw_decoder_hidden, out_decoder_hidden, content, scores, copy_to_ext, self.fields)
            all_prob[:, :, UNK] = -float('inf')
            # all_prob[:, :, fields['tgt'].vocab.__len__()] = -float('inf')
            inp = torch.max(all_prob, -1)[1]
            assert inp.size()[0] == 1 and inp.size()[1] == batch_size
            topk = all_prob.topk(10, -1)[1]
            dec_list.append(inp.clone().squeeze().data.cpu().numpy())
            dec_topk.append(topk.clone().squeeze().data.cpu().numpy())
        dec_list = np.array(dec_list).transpose(1, 0)
        dec_topk = np.array(dec_topk).transpose(1, 0, 2)
        return dec_list, dec_topk

    @staticmethod
    def recover_target_token(config, fields, pred_list, pred):
        # pred_list is the list with size (batch*step) and the elements is a int number
        r_lists = []
        if pred:
            step = config.max_dec_step
        else:
            step = pred_list.size()[1]
        if isinstance(pred_list, list):
            batch_step = pred_list.__len__()
        elif isinstance(pred_list, torch.Tensor):
            batch_step = pred_list.size()[0]
        else:
            raise TypeError('wrong type of pred_list')
        # print(batch_step)
        for i in range(batch_step):
            # filter topk results using layout information
            r_list = []
            for j in range(step):
                if pred_list[i][j] < fields['tgt'].vocab.__len__():
                    tk = fields['tgt'].vocab.itos[pred_list[i][j]]
                else:
                    tk = fields['copy_to_ext'].vocab.itos[pred_list[i][j] - fields['tgt'].vocab.__len__()]
                r_list.append(tk)

                if r_list[-1] == EOS_WORD:
                    r_list = r_list[:-1]
                    break
            r_lists.append(r_list)
        return r_lists

    def translator(self, batch):
        query, query_len = batch.src
        tgt = batch.tgt
        idxs = batch.indices.data.cpu().numpy().tolist()
        slen, batch_size = query.size()
        self.model.decoder.apply_mask(query)
        encoder_outputs, encoder_hidden_state = self.model.encoder(query, query_len)
        dec_list, dec_topk = self.run_decoder(encoder_outputs, encoder_hidden_state, batch_size, query_len, batch.copy_to_ext)
        dec_list = list(zip(idxs, dec_list.tolist()))
        dec_topk = list(zip(idxs, dec_topk.tolist()))
        return dec_list, dec_topk

    @staticmethod
    def save_results(config, pred_list, tgt_list, sv_result_path):
        with open(sv_result_path, 'w') as file:
            for pred, tgt in zip(pred_list, tgt_list):
                file.write(' '.join(tgt)+'\n')
                file.write(' '.join(pred)+'\n')
                file.write('\n')
        print('save done!')
    def get_config(self):
        return self.config

if __name__ == '__main__':
    test_epoch = 20
    checkpoint = '../data/django/model/model_%d.pt' % test_epoch
    g = Generator(None, checkpoint)
    config = g.get_config()
    train, dev, test, fields = built_dataset_vocab(config, config.train_js_path, config.dev_js_path,
                                                   config.test_js_path)
    test_js = read_anno_json(config.test_js_path)
    # fields = load_fields(train, dev, checkpoint)
    train.fields = fields
    dev.fields = fields
    test.fields = fields

    test_iter = OrderedIterator(test, config.batch_size, train=False,
                               repeat=False, sort=False, sort_within_batch=True)
    pred_results, tgt_results = [], []
    for batch in test_iter:
        pred_list, tgt_list = g.translator(batch)
        pred_results += pred_list
        tgt_results += tgt_list
    pred_results = sorted(pred_results, key=lambda x: x[0])
    pred_results = [p[1] for p in pred_results]
    gold_results = [i['tgt'] for i in test_js if i['tgt'].__len__()<=100]
    # print(pred_results)
    pred_results = g.recover_target_token(config, fields, pred_results, True)
    # print(pred_results)
    # print(gold_results)
    correct_rate = count_accuracy_int(pred_results, gold_results)
    print('test acc is: ', correct_rate)
    g.save_results(config, pred_results, gold_results, os.path.join(config.django_data_path, 'result/result_1202.txt'))
