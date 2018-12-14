import os, random
import numpy as np
import torch
### set random seed for everything
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

### param for django
class django_param:
    def __init__(self):
        #   data file para
        root_path = os.path.abspath('../')
        self.django_data_path = django_data_path = os.path.join(root_path, 'data/django')
        self.raw_django_anno_path = os.path.join(django_data_path, 'ase15-django-dataset-master/django/all.anno')
        self.raw_django_code_path = os.path.join(django_data_path, 'ase15-django-dataset-master/django/all.code')
        self.preprocess_django_anno_path_v1 = os.path.join(django_data_path, 'preprocess/anno_v1.txt')
        self.preprocess_django_code_path_v1 = os.path.join(django_data_path, 'preprocess/code_v1.txt')
        self.train_js_path = os.path.join(django_data_path, 'train.json')
        self.dev_js_path = os.path.join(django_data_path, 'dev.json')
        self.test_js_path = os.path.join(django_data_path, 'test.json')

        #   dataset para
        self.src_words_min_frequency = 0
        self.tgt_words_min_frequency = 0
        self.batch_size = 32
        self.word_vec_size = 200
        self.glove_vec_path = os.path.join(os.path.join(django_data_path, 'glove'), 'glove.6B.200d.txt.pt')

        #   model param
        self.classifer = 'copy'
        self.encoder_type = 'bilstm'
        self.encoder_hidden_dim = 256
        self.decoder_hidden_dim = 256
        self.dropout = 0.3
        self.attn_type = 'mlp'
        self.attn_hidden_dim = 64
        self.criterion = 'cross_entropy'
        self.optimizer = 'Adam'
        self.learning_rate = 0.001

        #   sv train param
        self.max_epochs = 50
        self.save_after_epoch = 10
        self.train_from_sv = None
        self.save_path = os.path.join(django_data_path, 'model')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        #   pg train param
        self.gamma_in_pg = 0.5
        self.train_from_pg = 5

        #   test param
        self.max_dec_step = 100
