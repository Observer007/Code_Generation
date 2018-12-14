import torch
import tokenize
from io import BytesIO
import re, nltk
import numpy as np
import config as _config
t = torch.Tensor([[2],[3],[4], [4], [5], [6]])
a = torch.Tensor().new_ones(2, 3, 3)
print(t.squeeze(-1))
# print(t.unsqueeze(1).expand(2,3,-1))

a = 'else = _STR:0_ '
code = tokenize.tokenize(BytesIO(a.encode('utf-8')).readline)
print(list(code))

special_words = re.compile('([\'\"])abd\1')
a = '"abd" replaced by value "djfkldsjfklds"'
b = re.finditer(r'([\'\"]).*?\1', a, flags=0)
print(list(b))

x = torch.cuda.FloatTensor([0.1, 0.2, 0.7])
print(np.random.choice(range(3), p=x))


a = torch.Tensor([[1,2],[3,4],[5,6]])
b = torch.Tensor([[1,2],[3,4],[5,6]])
c = torch.Tensor([[1],[2],[3]])
print(torch.mul(a,b))
print(c.expand_as(a))
print(a*b)

t = 'abd \'dkfjldk\' d.fdkfjl'
print(nltk.word_tokenize(t))

x = np.array([[1, 2, 3]])
print(x.transpose([1, 0]))
print(np.repeat(np.expand_dims(x, -1), 3, axis=1))