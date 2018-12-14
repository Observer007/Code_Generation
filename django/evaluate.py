import ast
import re, sys
import numpy as np
import pyflakes
import torch

def typename(x):
    return type(x).__name__

def escape(text):
    text = text \
        .replace('"', '`') \
        .replace('\'', '`') \
        .replace(' ', '-SP-') \
        .replace('\t', '-TAB-') \
        .replace('\n', '-NL-') \
        .replace('(', '-LRB-') \
        .replace(')', '-RRB-') \
        .replace('|', '-BAR-')
    return repr(text)[1:-1] if text else '-NONE-'

def makestr(node):

    #if node is None or isinstance(node, ast.Pass):
    #    return ''

    if isinstance(node, ast.AST):
        n = 0
        nodename = typename(node)
        s = '(' + nodename
        for chname, chval in ast.iter_fields(node):
            chstr = makestr(chval)
            if chstr:
                s += ' (' + chname + ' ' + chstr + ')'
                n += 1
        if not n:
            s += ' -' + nodename + '-' # (Foo) -> (Foo -Foo-)
        s += ')'
        return s

    elif isinstance(node, list):
        n = 0
        s = '(list'
        for ch in node:
            chstr = makestr(ch)
            if chstr:
                s += ' ' + chstr
                n += 1
        s += ')'
        return s if n else ''

    elif isinstance(node, str):
        return '(str ' + escape(node) + ')'

    elif isinstance(node, bytes):
        return '(bytes ' + escape(str(node)) + ')'

    else:
        return '(' + typename(node) + ' ' + str(node) + ')'

def check_code_correctness(codes):
    p_elif = re.compile(r'^elif\s?')
    p_else = re.compile(r'^else\s?')
    p_try = re.compile(r'^try\s?')
    p_except = re.compile(r'^except\s?')
    p_finally = re.compile(r'^finally\s?')
    p_decorator = re.compile(r'^@.*')

    if isinstance(codes, str):
        codes = [codes]
    elif isinstance(codes, list):
        pass
    else:
        raise ValueError('wrong type of codes for check')

    result_list = []
    for l in codes:
        l = l.strip()
        if not l:
            # print(l)
            result_list.append(0)
            continue
        if p_elif.match(l): l = 'if True: pass\n' + l
        if p_else.match(l): l = 'if True: pass\n' + l

        if p_try.match(l):
            l = l + 'pass\nexcept: pass'
        elif p_except.match(l):
            l = 'try: pass\n' + l
        elif p_finally.match(l):
            l = 'try: pass\n' + l

        if p_decorator.match(l):
            # print('decorator:', l)
            l = l + '\ndef dummy(): pass'
        if l[-1] == ':': l = l + 'pass'
        try:
            parse = ast.parse(l)
            parse = parse.body[0]
            dump = makestr(parse)
            # print('ast:', dump)
            sys.stdout.flush()
            result_list.append(0.5)
        except:
            # print('wrong data:', l)
            result_list.append(0)
    results = np.array(result_list)
    results = np.expand_dims(results, 1)
    return results

def case_test():
    code = 'del getattr ( obj . name, self . name )'
    code = 'else : float ( 1.0 )'
    # code = 'print(__STR:0__)'
    check_code_correctness(code)
def file_test():
    codes = []
    with open('../data/django/test.txt', 'r') as file:
        idx = 0
        for line in file:
            if idx % 3 == 0:
                codes.append(line)
            idx += 1
    # with open('../data/django/ase15-django-dataset-master/django/all.code', 'r') as file:
    #     for line in file:
    #         codes.append(line)
    result_list = check_code_correctness(codes)
    result_list = np.array(result_list)
    print(np.sum(result_list == 1)/len(result_list))
def count_accuracy_int(pred_list, tgt_list):
    assert len(pred_list) == len(tgt_list)
    correct_count = 0
    for pred, tgt in zip(pred_list, tgt_list):
        flag = True
        for i in range(len(tgt)):
            if i >= pred.__len__() or pred[i] != tgt[i]:
                flag = False
                break
        if flag:
            correct_count += 1
    return correct_count / len(tgt_list)
def count_accuracy(scores, target, mask=None, row=False):
    if scores.dim() == 3:
        pred = scores.max(2)[1]
    elif scores.dim() == 2:
        pred = scores
    else:
        raise ValueError('wrong dim of scores')
    if pred.size()[0]>target.size()[0]:
        pred = pred[:target.size()[0], :]
    if mask is None:
        # m_correct = pred.eq(target)
        # num_all = m_correct.numel()
        raise ValueError('miss mask value')
    elif row:
        m_correct = pred.eq(target).masked_fill_(
            mask, 1).prod(0, keepdim=False)
        m_correct_list = pred.data.eq(target).masked_fill_(
            mask, 1)
        num_all = m_correct.numel()
    else:
        non_mask = mask.ne(1)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum()
    return (m_correct.cpu().numpy(), num_all, m_correct_list)
if __name__ == '__main__':
    case_test()
    # file_test()