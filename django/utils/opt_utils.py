import torch

def onehot(index, N=None, ignore_index=None):
    """
    return a onehot torch tensor with index=index
    :param index:
    :param ignore_index:
    :return:
    """
    size_list = list(index.size())
    if N is None:
        N = index.max()+1
    onehot_tensor = torch.zeros(*size_list, N).scatter_(-1, index.unsqueeze(-1), 1)
    onehot_tensor.masked_fill_(index.data.eq(ignore_index).unsqueeze(-1), 0)
    return onehot_tensor.cuda()