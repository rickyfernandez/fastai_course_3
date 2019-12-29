import time
import torch
from torch import tensor
from functools import partial

from exp.utils import listify
from exp.optimizer import SgdOptimizer
from exp.callbacks import TrainEvalCallback, CancelTrainException, CancelEpochException, CancelBatchException

def param_getter(m): return m.parameters()

class Learner:
    """
    Class that holds all components to train a model.
    """
    def __init__(self, model, data, loss_func, opt_func=SgdOptimizer,
            lr=1e-2, splitter=param_getter, cbs=None, cb_funcs=None):

        self.model, self.data, self.loss_func = model, data, loss_func
        self.opt_func, self.lr, self.splitter = opt_func, lr, splitter
        self.in_train, self.logger, self.opt = False, print, None

        self.epoch, self.data_loader = None, None
        self.iter, self.xb, self.yb = None, None, None
        self.pred, self.loss, self.train = None, None, None

        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def add_cbs(self, cbs):
        """Add list of callbacks as an attribute"""
        for cb in listify(cbs): self.add_cb(cb)

    def add_cb(self, cb):
        """Add callback as an attribute with reference to learner"""
        cb.set_learner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        """Remove callback"""
        for cb in listify(cbs): self.cbs.remove(cb)

    def one_batch(self, i, xb, yb):
        """Run through one batch in model for training or prediction"""
        try:

            self.iter = i
            self.xb, self.yb = xb, yb;                      self("begin_batch")
            self.pred = self.model(self.xb);                self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb); self("after_loss")

            if not self.in_train: return

            self.loss.backward();                           self("after_backward")
            self.opt.step();                                self("after_step")
            self.opt.zero_grad()

        except CancelBatchException:                        self("after_cancel_batch")
        finally:                                            self("after_batch")

    def all_batches(self):
        """Run through all batches in data set"""
        self.iters = len(self.data_loader)
        try:
            for i, (xb,yb) in enumerate(self.data_loader):
                self.one_batch(i, xb, yb)
        except CancelEpochException: self("after_cancel_epoch")

    def do_begin_fit(self, epochs):
        """First call before fit is called"""
        self.epochs, self.loss = epochs, tensor(0.)
        self("begin_fit")

    def do_begin_epoch(self, epoch):
        """First call before begin epoch called"""
        self.epoch, self.data_loader = epoch, self.data.train_dl
        #return self("begin_epoch")

    def fit(self, epochs, cbs=None, reset_opt=False):
        """Fit model on data"""
        self.add_cbs(cbs)
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):

                self.do_begin_epoch(epoch)
                if not self("begin_epoch"): self.all_batches()

                with torch.no_grad():
                    self.data_loader = self.data.valid_dl
                    if not self("begin_validate"): self.all_batches()

                self("after_epoch")

        except CancelTrainException: self("after_cancel_train")
        finally:
            self("after_fit")
            self.remove_cbs(cbs)

    ALL_CBS = {"begin_batch", "after_pred", "after_loss", "after_backward", "after_step",
        "after_cancel_batch", "after_batch", "after_cancel_epoch", "begin_fit",
        "begin_epoch", "begin_validate", "after_epoch",
        "after_cancel_train", "after_fit"}

    def __call__(self, cb_name):
        """Call every callback registered"""
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res
