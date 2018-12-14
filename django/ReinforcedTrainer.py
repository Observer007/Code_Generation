import torch
import torch.nn as nn
import torchtext
import os
from utils.data_utils import *
from evaluate import count_accuracy, check_code_correctness
from model import LinearClassifer, CopyNetwork
from generator import Generator
from utils.gpu_mem_track import MemTracker
import inspect
device = torch.device('cuda:1')

frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame)         # define a GPU tracker
class PolicyGradientTrainer:
    def __init__(self, model, fields, train_iter, val_iter, config):
        self.model = model
        self.fields = fields
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.config = config
        if config.train_from_pg > 0:
            checkpoint = '../data/django/model/model_%d.pt' % config.train_from_pg
            c = torch.load(checkpoint, map_location=lambda x, y: x)
            self.model.load_state_dict(c['model'])
        if config.criterion == 'cross_entropy':
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=PAD)
        else:
            raise ValueError('wrong criterion type')
        self.params = [i for i in self.model.parameters() if i.requires_grad]
        if config.optimizer == 'Adam':
            self.optimizor = torch.optim.Adam(self.params, lr=config.learning_rate,
                                              betas=[0.9, 0.98], eps=1e-9)
        if isinstance(self.model.classifer, CopyNetwork):
            self.action_num = self.fields['tgt'].vocab.__len__() + self.fields['copy_to_ext'].vocab.__len__()
        elif isinstance(self.classifer, LinearClassifer):
            self.action_num = self.fields['tgt'].vocab.__len__()
        self.discount = config.gamma_in_pg
        self.actions = []
        self.rewards = []
        self.states  = []
        self.model.train()

    def choose_action(self, action_probs):
        assert action_probs.dim() == 2
        batch, action_num = action_probs.size()
        choice_actions = []
        for i in range(batch):
            action = torch.multinomial(torch.exp(action_probs[i]), 1, replacement=True)
            choice_actions.append(action)
        return torch.stack(choice_actions).squeeze()

    def _discount_rewards(self, rewards, mask_len=None):
        # consider rewards as a tensor with size (tlen* batch)
        rewards = rewards.transpose(0, 1)
        discount_reward = np.zeros_like(rewards)
        for b in range(rewards.size()[0]):
            running_add = 0
            if mask_len is None:
                instance_len = rewards.size()[1]
            else:
                instance_len = mask_len[b] - 1
            for i in reversed(range(instance_len)):
                if rewards[b][i].ge(1e-3):
                    running_add = 0
                running_add = running_add * self.discount + rewards[b][i]
                discount_reward[b][i] = running_add
            # for i in range(instance_len, rewards.size()[1]):
            #     discount_reward[b][i] = 0
        # print(discount_reward)
        # discount_reward = discount_reward - np.repeat(np.expand_dims(np.mean(discount_reward, axis=-1), -1), discount_reward.shape[1], axis=-1)
        # discount_reward = discount_reward / (1e-6+np.repeat(np.expand_dims(np.std(discount_reward, axis=-1), -1), discount_reward.shape[1], axis=-1))
        return torch.cuda.FloatTensor(discount_reward.transpose([1, 0]))

    def cal_loss(self, logit, label, reward, mask=None):
        loss_sum = []
        for i, (pred, true, r, m) in enumerate(zip(logit, label, reward, mask)):
            # if mask is not None:
            #     true = torch.mul(true, mask.long())
            pred = pred.squeeze()
            loss = 0
            for j in range(pred.size()[0]):                 # batch
                loss += -pred[j][true[j]]*m[j]*r[j]
            loss_sum.append(loss/pred.size()[0])
        return sum(loss_sum)

    def forward(self, epoch, batch, fields):
        # ================== forward propagation =========================================
        query, query_len = batch.src
        query, query_len = query.cuda(), query_len.cuda()
        tgt, tgt_len = batch.tgt
        tgt_gpu, tgt_len_gpu = tgt.cuda(), tgt_len.cuda()
        copy_to_ext = batch.copy_to_ext

        self.model.decoder.apply_mask(query)

        tgt_input = tgt_gpu[0].unsqueeze(0)
        total_loss = []
        encoder_outputs, encoder_hidden_state = self.model.encoder(query, query_len)
        decoder_hidden_state_tuple = [torch.cat([encoder_hidden_state[0][0], encoder_hidden_state[0][1]], 1),
                                     torch.cat([encoder_hidden_state[1][0], encoder_hidden_state[1][1]], 1)]
        content, init_content = None, None
        tgt_output = []
        actions = []
        for i in range(torch.max(tgt_len)-1):
            if i>0:
                init_content = content
            # gpu_tracker.track()
            tgt_input.data.masked_fill_(tgt_input.data.ge(fields['tgt'].vocab.__len__()), UNK)
            # gpu_tracker.track()
            raw_decoder_hidden, out_decoder_hidden, scores, content, decoder_hidden_state_tuple \
                = self.model.decoder(tgt_input, encoder_outputs, query_len, decoder_hidden_state_tuple, init_content)
            # gpu_tracker.track()
            all_prob, copy_scores = self.model.classifer(raw_decoder_hidden, out_decoder_hidden, content, scores,
                                                         copy_to_ext, self.fields)
            # gpu_tracker.track()
            tlen, batch_, outputdim = all_prob.size()
            assert all_prob.size()[2] == fields['tgt'].vocab.__len__() + fields['copy_to_ext'].vocab.__len__() and tlen==1
            # ================== choose actions ==============================================
            all_prob[:, :, UNK] = -float('inf')
            # gpu_tracker.track()
            action = self.choose_action(all_prob.squeeze().detach())            # batch*word dim
            # print(action.size())
            # gpu_tracker.track()
            tgt_input = action.unsqueeze(0).clone()
            tgt_output.append(all_prob)
            actions.append(action)
            # gpu_tracker.track()
        actions_tensor = torch.stack(actions)
        tgt_output = torch.stack(tgt_output)
        # print(self.model.parameters)
        # ================== calculate rewards ===============================================
        if isinstance(self.model.classifer, CopyNetwork):
            tgt_copy_mask = batch.tgt_copy_ext.ne(fields['tgt_copy_ext'].vocab.stoi[UNK_WORD]).long()[1:]
            tgt_gen_mask = batch.tgt_copy_ext.eq(fields['tgt_copy_ext'].vocab.stoi[UNK_WORD]).long()[1:]
            tgt_label = torch.mul(tgt_copy_mask, batch.tgt_copy_ext[1:] + fields['tgt'].vocab.__len__()) +\
            torch.mul(tgt_gen_mask, tgt[1:])
        elif isinstance(self.model.classifer, LinearClassifer):
            tgt_label = tgt[1:]
        else:
            raise TypeError('wrong classifer')
        correct_m_and_count = count_accuracy(actions_tensor, target=tgt_label.cuda(),
                                             mask=tgt_label.data.eq(PAD).cuda(), row=True)
        rewards = torch.cuda.FloatTensor(correct_m_and_count[0]).expand_as(correct_m_and_count[-1])

        # =========================== add rule rewards =====================================
        # token_lists = Generator.recover_target_token(self.config, self.fields, actions_tensor.transpose(0, 1), pred=False)
        # token_lists = [' '.join(token_list) for token_list in token_lists]
        # acc_rewards = check_code_correctness(token_lists)
        # rewards = rewards.float() + torch.cuda.FloatTensor(acc_rewards).squeeze().unsqueeze(0).expand_as(rewards)
        rewards = self._discount_rewards(rewards, mask_len=tgt_len_gpu)

        # ================== calculate loss ==============================================
        # print(tgt_label.ne(PAD).float().size())
        loss = self.cal_loss(tgt_output, actions, rewards, tgt_label.data.ne(PAD).float().cuda())
        # gpu_tracker.track()
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
        torch.save(check_point, os.path.join(config.save_path, 'pg_model_%s.pt' % epoch))