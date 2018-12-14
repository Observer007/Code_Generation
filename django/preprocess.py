import tokenize
import torch
import torchtext
from io import BytesIO
import numpy as np

from nltk.tokenize import word_tokenize
from utils.data_utils import TableDataset
import re,json, codecs, os
import nltk
import config as config_

type_dict = {1:'NAME', 53:'OP', 3:'STRING', 2:'NUMBER'}
def preprocess_anno_nltk(file_path, new_file_path):
    lines = []
    pre_line = ''
    idx = 0
    error_num = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            idx += 1
            if len(pre_line) != 0:
                error_num += 1
                line = line.split('   ')
                # print(idx, line)
                if len(line) == 1:
                    new_line1 = pre_line.split(' ')
                    new_line1 = [i for i in new_line1 if len(i) > 0]
                    new_line2 = line[0].split(' ')
                    new_line2 = [i for i in new_line2 if len(i) > 0]
                    pre_line = ''
                    lines.append(new_line1)
                    lines.append(new_line2)
                elif len(line) == 2:
                    new_line1 = (pre_line + line[0]).split(' ')
                    new_line1 = [i for i in new_line1 if len(i) > 0]
                    new_line2 = (line[1]).split(' ')
                    new_line2 = [i for i in new_line2 if len(i) > 0]
                    pre_line = ''
                    lines.append(new_line1)
                    lines.append(new_line2)
                else:
                    # print(idx, len(line), line)
                    flag = True
                    for i in range(1, len(line) - 1):
                        if line[i][-1] == '.':
                            flag = False
                            new_line1 = (pre_line + ' '.join(line[:i + 1])).split(' ')
                            new_line1 = [i for i in new_line1 if len(i) > 0]
                            pre_line = ' '.join(line[i + 1:])
                            lines.append(new_line1)
                            break
                    if flag:
                        new_line1 = (pre_line + ' '.join(line[:-1])).split(' ')
                        new_line1 = [i for i in new_line1 if len(i) > 0]
                        pre_line = line[-1]
                        lines.append(new_line1)
                continue
                # for i in range(len(line[1:-1])):
                #     if line[i+1][0] == ' ':
                #         new_line1 = pre_line + '.'.join(line[:i+1])
                #         lines.append(new_line1.split(' '))
                #         lines.append('.'.join(line[i+1]).strip().split(' '))
            if line[-1] == '.':
                line = line.split(' ')
                line = [i for i in line if len(i) > 0]
                lines.append(line)
                pre_line = ''
            else:
                pre_line = line
    print('error rate:', error_num / idx)

    # split a word with '.' inside
    new_lines = []
    for line in lines:
        new_line = []
        for word in line:
            word_list = nltk.word_tokenize(word)
            if len(word_list) == 1:
                new_line.append(word)
            else:
                new_line += word_list
        new_lines.append(new_line)
    with open(new_file_path, 'w') as file:
        for line in new_lines:
            file.write(' '.join(line) + '\n')


def preprocess_anno_manual(file_path, new_file_path):
    lines = []
    pre_line = ''
    idx = 0
    error_num = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            idx += 1
            if len(pre_line) != 0:
                error_num += 1
                line = line.split('   ')
                # print(idx, line)
                if len(line) == 1:
                    new_line1 = pre_line.split(' ')
                    new_line1 = [i for i in new_line1 if len(i) > 0]
                    new_line2 = line[0].split(' ')
                    new_line2 = [i for i in new_line2 if len(i) > 0]
                    pre_line = ''
                    lines.append(new_line1)
                    lines.append(new_line2)
                elif len(line) == 2:
                    new_line1 = (pre_line+line[0]).split(' ')
                    new_line1 = [i for i in new_line1 if len(i) > 0]
                    new_line2 = (line[1]).split(' ')
                    new_line2 = [i for i in new_line2 if len(i) > 0]
                    pre_line = ''
                    lines.append(new_line1)
                    lines.append(new_line2)
                else:
                    # print(idx, len(line), line)
                    flag = True
                    for i in range(1, len(line)-1):
                        if line[i][-1] == '.':
                            flag = False
                            new_line1 = (pre_line+' '.join(line[:i+1])).split(' ')
                            new_line1 = [i for i in new_line1 if len(i) > 0]
                            pre_line = ' '.join(line[i+1:])
                            lines.append(new_line1)
                            break
                    if flag:
                        new_line1 = (pre_line+' '.join(line[:-1])).split(' ')
                        new_line1 = [i for i in new_line1 if len(i)>0]
                        pre_line = line[-1]
                        lines.append(new_line1)
                continue
                # for i in range(len(line[1:-1])):
                #     if line[i+1][0] == ' ':
                #         new_line1 = pre_line + '.'.join(line[:i+1])
                #         lines.append(new_line1.split(' '))
                #         lines.append('.'.join(line[i+1]).strip().split(' '))
            if line[-1] == '.':
                line = line.split(' ')
                line = [i for i in line if len(i) > 0]
                lines.append(line)
                pre_line = ''
            else:
                pre_line = line
    print('error rate:', error_num/idx)

    # split a word with '.' inside
    new_lines = []
    for line in lines:
        new_line = []
        for word in line:
            if len(word.split('.'))==1:
                new_line.append(word)
            else:
                blank_count = 0
                valid_list = []
                for word_split in word.split('.'):
                    if len(word_split) != 0:
                        blank_count+=1
                        valid_list.append(word_split)
                if blank_count>1:
                    tmp_list = ['(']+valid_list+[')']
                    new_line.append(word)
                    new_line += tmp_list
                else:
                    new_line.append(word)
        new_lines.append(new_line)
    # =========================== split word with ',' or '.' end ========================
    new_lines_ = []
    for line in new_lines:
        new_line_ = []
        for word in line:
            # drop empty word
            if len(word) == 0:
                continue
            if not (word[-1].isdigit() or word[-1].isalnum()) and word[-1] not in ['\'', '\"']:
                if len(word) == 1:
                    new_line_.append(word)
                else:
                    new_line_ += [word[:-1], word[-1]]
            else:
                new_line_.append(word)
        new_lines_.append(new_line_)
    with open(new_file_path, 'w') as file:
        for line in new_lines_:
            file.write(' '.join(line)+'\n')

def preprocess_code(code_path, new_code_path):
    codes = []
    types = []
    with open(code_path, 'r') as file:
        for code in file.readlines():
            code = code.strip()
            code = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
            # print(list(code))
            one_code = []
            one_type = []
            for item in code:
                if len(item.line)>0:
                    one_code.append(item.string)
                    one_type.append(type_dict[item.type])

            codes.append(one_code)
            types.append(one_type)
    with open(new_code_path, 'w') as file:
        for code, type in zip(codes, types):
            file.write(' '.join(code)+'$$$')
            file.write(' '.join(type)+'\n')

def replace_special(anno, code):
    special_words = re.compile(r"([\'\"]).*?\1")
    anno_list = list(special_words.finditer(anno))
    code_list = list(special_words.finditer(code))
    anno_list = [i.group() for i in anno_list]
    code_list = [i.group() for i in code_list]
    for i, match in enumerate(anno_list):
        match_list = match[1:-1].split(' ')
        if len(match_list) == 1:
            anno = anno.replace(match, '_STR:%d_' % i)
        else:
            anno = anno.replace(match, ' '.join(match_list))
    for match in code_list:
        if match in anno_list:
            match_list = match[1:-1].split(' ')
            if len(match_list) == 1:
                count = anno_list.index(match)
                code = code.replace(match, '_STR:%d_' % count)
            else:
                code = code.replace(match, ' '.join(['\"'] + match_list+ ['\"']))
    for i, match in enumerate(anno_list):
        if match in code:
            count = anno_list.index(match)
            code = code.replace(match, '_STR:%d_' % count)
    return anno, code

def merge_examples(anno_path, code_path, train_path, dev_path, test_path):
    with open(anno_path, 'r') as file:
        annos = file.readlines()
    with open(code_path, 'r') as file:
        codes_and_types = file.readlines()
    examples = []
    idx = 0
    js_dict = []

    for anno, token in zip(annos, codes_and_types):
        # if idx==17886:
        #     d = 1
        anno = anno.strip()
        token = token.strip()
        # print(token)
        [code, type] = token.split('$$$')
        type = type.split(' ')
        anno, code = replace_special(anno, code)
        js_dict = {'src': anno.split(' '), 'tgt': code.split(' '), 'type': type}
        tmp_str = json.dumps(js_dict)+'\n'
        if idx<16000:
            with codecs.open(train_path, 'a', 'utf-8') as file:
                file.write(tmp_str)
        elif idx<17000:
            with codecs.open(dev_path, 'a', 'utf-8') as file:
                file.write(tmp_str)
        else:
            with codecs.open(test_path, 'a', 'utf-8') as file:
                file.write(tmp_str)
        idx += 1



def read_anno_json(anno_path):
    with codecs.open(anno_path, "r", "utf-8") as corpus_file:
        js_list = [json.loads(line) for line in corpus_file]
    return js_list

def built_dataset_vocab(config, train_js_path, dev_js_path, test_js_path, save=False):
    fields = TableDataset.get_fields()
    train_js = read_anno_json(train_js_path)
    dev_js = read_anno_json(dev_js_path)
    test_js = read_anno_json(test_js_path)

    train = TableDataset(train_js, fields, None, None, True)
    dev = TableDataset(dev_js, fields, None, None, True)
    test = TableDataset(test_js, fields, None, None, True)

    TableDataset.build_vocab(train, dev, test, config)
    TableDataset.print(train.fields)
    if save:
        torch.save(TableDataset.save_vocab(fields), open(os.path.join(config.django_data_path, 'vocab.pt'), 'wb'))
        torch.save(train, open(os.path.join(config.django_data_path, 'train.pt'), 'wb'))
        torch.save(dev, open(os.path.join(config.django_data_path, 'dev.pt'), 'wb'))
        torch.save(test, open(os.path.join(config.django_data_path, 'test.pt'), 'wb'))
    return train, dev, test, fields
if __name__ == '__main__':
    config = config_.django_param()
    preprocess_anno_manual(config.raw_django_anno_path, config.preprocess_django_anno_path_v1)
    preprocess_code(config.raw_django_code_path, config.preprocess_django_code_path_v1)
    merge_examples(config.preprocess_django_anno_path_v1, config.preprocess_django_code_path_v1, config.train_js_path, config.dev_js_path, config.test_js_path)
    # built_dataset_vocab(config.train_js_path, config.dev_js_path, config.test_js_path)