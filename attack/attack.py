import torch
class Attack(object):
    def __init__(self,name,args):
        self.name = name
        self.args = args

    def forward(self, *input):
        raise NotImplementedError
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)