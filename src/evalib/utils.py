import torch


def get_device():
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print('Cuda available?', has_cuda)

    return device
