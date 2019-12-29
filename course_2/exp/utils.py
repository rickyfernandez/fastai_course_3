import math
import torch
import os, mimetypes, re
from typing import Iterable

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')

def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def listify(obj):
    if obj is None: return []
    if isinstance(obj, list): return obj
    if isinstance(obj, str): return [obj]
    if isinstance(obj, Iterable): return list(obj)
    return [obj]

def accuracy(out, yb):
    return (torch.argmax(out, dim=1)==yb).float().mean()

#def setify(o): return o if isinstance(o, set) else set(listify(o))
#
#def _get_files(p, fs, extensions=None):
#    p = Path(p)
#    res = [p/f for f in fs if not f.startswith('.')
#           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
#    return res
#
#def get_files(path, extensions=None, recurse=False, include=None):
#    path = Path(path)
#    extensions = setify(extensions)
#    extensions = {e.lower() for e in extensions}
#    if recurse:
#        res = []
#        for i, (p, d, f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
#            if include is not None and i==0: d[:] = [o for o in d if o in include]
#            else:                            d[:] = [o for o in d if not o.startswith('.')]
#            res += _get_files(p, f, extensions)
#        return res
#    else:
#        f = [o.name for o in os.scandir(path) if o.is_file()]
#        return _get_files(path, f, extensions)
#
def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

#class ItemList(ListContainer):
#    def __init__(self, items, path=".", tfms=None):
#        super().__init__(items)
#        self.path, self.tfms = Path(path), tfms
#
#    def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'
#
#    def new(self, items, cls=None):
#        if cls is None: cls=self.__class__
#        return cls(items, self.path, tfms=self.tfms)
#
#    def get(self, i): return i
#    def _get(self, i): return compose(self.get(i), self.tfms)
#
#    def __getitem__(self, idx):
#        res = super().__getitem__(idx)
#        if isinstance(res, list): return [self._get(o) for o in res]
#        return self._get(res)

def prev_pow_2(x): return 2**math.floor(math.log2(x))

def model_summary(run, learn, data, find_all=False):
    xb,yb = get_batch(data.valid_dl, run)
    device = next(learn.model.parameters()).device#Model may not be on the GPU yet
    xb,yb = xb.to(device),yb.to(device)
    mods = find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    f = lambda hook,mod,inp,out: print(f"{mod}\n{out.shape}\n")
    with Hooks(mods, f) as hooks: learn.model(xb)
