import numpy as np
import pandas as pd
import torch
import torchtext
from itertools import chain
from collections import defaultdict, Counter


UNK_WORD = '<unk>'
UNK = 0
PAD_WORD = '<blank>'
PAD = 1
BOS_WORD = '<s>'
BOS = 2
EOS_WORD = '</s>'
EOS = 3
special_token_list = [UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD]


def get_parent_index(tk_list):
    stack = [0]
    r_list = []
    for i, tk in enumerate(tk_list):
        r_list.append(stack[-1])
        if tk.startswith('('):
            # +1: because the parent of the top level is 0
            stack.append(i+1)
        elif tk ==')':
            stack.pop()
    # for EOS (</s>)
    r_list.append(0)
    return r_list


def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class TableDataset(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return len(ex.src)

    def __init__(self, anno, fields, permute_order, opt, filter_ex, **kwargs):
        """
        Create a TranslationDataset given paths and fields.

        anno: location of annotated data / js_list
        filter_ex: False - keep all the examples for evaluation (should not have filtered examples); True - filter examples with unmatched spans;
        """
        if isinstance(anno, str):
            # TODO change the read function
            # js_list = read_anno_json(anno, opt)
            js_list = None
            pass
        else:
            js_list = anno
        js_list = [i for i in js_list if len(i['src'])<=100]
        js_list = [i for i in js_list if len(i['tgt'])<=100]
        src_data = self._read_annotated_file(opt, js_list, 'src', filter_ex)
        src_examples = self._construct_examples(src_data, 'src')

        copy_to_tgt_data = self._read_annotated_file(
            opt, js_list, 'copy_to_tgt', filter_ex)
        copy_to_tgt_examples = self._construct_examples(
            copy_to_tgt_data, 'copy_to_tgt')

        copy_to_ext_data = self._read_annotated_file(
            opt, js_list, 'copy_to_ext', filter_ex)
        copy_to_ext_examples = self._construct_examples(
            copy_to_ext_data, 'copy_to_ext')

        tgt_data = self._read_annotated_file(opt, js_list, 'tgt', filter_ex)
        tgt_examples = self._construct_examples(tgt_data, 'tgt')


        tgt_copy_ext_data = self._read_annotated_file(
            opt, js_list, 'tgt_copy_ext', filter_ex)
        tgt_copy_ext_examples = self._construct_examples(tgt_copy_ext_data, 'tgt_copy_ext')

        # examples: one for each src line or (src, tgt) line pair.
        examples = [join_dicts(*it) for it in
                    zip(src_examples, copy_to_tgt_examples, copy_to_ext_examples, tgt_examples,
                    tgt_copy_ext_examples)]
        # the examples should not contain None
        len_before_filter = len(examples)
        examples = list(filter(lambda x: all(
            (v is not None for k, v in x.items())), examples))
        len_after_filter = len(examples)
        num_filter = len_before_filter - len_after_filter
        if num_filter > 0:
            print('Filter #examples (with None): {} / {} = {:.2%}'.format(num_filter,
                                                                          len_before_filter,
                                                                          num_filter / len_before_filter))

        # Peek at the first to see which fields are used.
        ex = examples[0]
        keys = ex.keys()
        fields = [(k, fields[k])
                  for k in (list(keys) + ["indices"])]

        def construct_final(examples):
            f = []
            for i, ex in enumerate(examples):
                f.append(torchtext.data.Example.fromlist(
                    [ex[k] for k in keys] + [i],
                    fields))
            return f

        # def filter_pred(example):
        #     return True

        super(TableDataset, self).__init__(
            construct_final(examples), fields, None)

    def _read_annotated_file(self, opt, js_list, field, filter_ex):
        """
        path: location of a src or tgt file
        truncate: maximum sequence length (0 for unlimited)
        """
        if field in ('src'):
            lines = (line[field] for line in js_list)
        elif field in ('copy_to_tgt', 'copy_to_ext'):
            lines = (line['src'] for line in js_list)
        elif field in ('tgt',):
            lines = (line['tgt'] for line in js_list)
        elif field in ('tgt_copy_ext',):
            def _tgt_copy_ext(line):
                r_list = []
                src_set = set(line['src'])
                for tk_tgt in line['tgt']:
                    if tk_tgt in src_set:
                        r_list.append(tk_tgt)
                    else:
                        r_list.append(UNK_WORD)
                return r_list

            lines = (_tgt_copy_ext(line) for line in js_list)
        else:
            raise NotImplementedError
        for line in lines:
            yield line

    def _construct_examples(self, lines, side):
        for words in lines:
            example_dict = {side: words}
            yield example_dict

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(TableDataset, self).__reduce_ex__()

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = TableDataset.get_fields()
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def get_fields():
        fields = {}
        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)
        fields["copy_to_tgt"] = torchtext.data.Field(pad_token=UNK_WORD)
        fields["copy_to_ext"] = torchtext.data.Field(pad_token=UNK_WORD)
        fields["tgt"] = torchtext.data.Field(include_lengths=True,
            init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)
        fields["tgt_copy_ext"] = torchtext.data.Field(
            init_token=UNK_WORD, eos_token=UNK_WORD, pad_token=UNK_WORD)
        fields["indices"] = torchtext.data.Field(
            use_vocab=False, sequential=False)
        return fields

    @staticmethod
    def build_vocab(train, dev, test, opt):
        fields = train.fields
        src_vocab_all = []
        # build vocabulary only based on the training set
        # the last one should be the variable 'train'
        for split in (dev, test, train,):
            fields['src'].build_vocab(split, min_freq=0)
            src_vocab_all.extend(list(fields['src'].vocab.stoi.keys()))

        # build vocabulary only based on the training set
        for field_name in ['src']:
            fields[field_name].build_vocab(
                train, min_freq=opt.src_words_min_frequency)

        # build vocabulary only based on the training set
        for field_name in ['tgt']:
            fields[field_name].build_vocab(
                train, min_freq=opt.tgt_words_min_frequency)

        fields['copy_to_tgt'].vocab = fields['tgt'].vocab
        # fields['src_all'].vocab - fields['tgt'].vocab
        cnt_ext = Counter()
        for k in src_vocab_all:
            if k not in fields['tgt'].vocab.stoi:
                cnt_ext[k] = 1
        fields['copy_to_ext'].vocab = torchtext.vocab.Vocab(cnt_ext, specials=list(special_token_list), min_freq=0)
        fields['tgt_copy_ext'].vocab = fields['copy_to_ext'].vocab
    @staticmethod
    def print(fields):
        assert isinstance(fields, dict)
        for key in fields:
            if hasattr(fields[key], 'vocab'):
                print('%s has %d words' % (key, fields[key].vocab.__len__()))
