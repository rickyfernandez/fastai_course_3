
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/02_fully_connected.ipynb

from exp.nb_01 import *

def get_data():
    path = datasets.download_data(
        MNIST_URL,
        fname="/home/ricky/Desktop/repos/fastai_course/data/mnist/mnist.pkl.gz",
        ext=".gz")
    with gzip.open(path,"rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train, y_train, x_valid, y_valid))

def normalize(x, m, s): return (x-m)/s

def test_near_zero(a, tol=1e-3): assert a.abs() < tol, f"Near zero: {a}"

from torch.nn import init

from torch import nn