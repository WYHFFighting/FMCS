from torch.optim.lr_scheduler import _LRScheduler
import math

class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        # ******************
        # polynomialDecayLr
        # ******************
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        # self.meanFunc_lr = lr[0]
        # self.lr = lr[1]
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
            # meanFunc_lr = self.warmup_factor * self.meanFunc_lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr

        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


class CosineDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates_prop, tot_updates, lr, lrf, view, mea_func_lr, last_epoch=-1, verbose=False):
        self.warmup_updates = int(tot_updates * warmup_updates_prop)
        # self.tot_updates = tot_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.lrf = lrf
        self.mean_func_lr = mea_func_lr
        self.view = view
        super(CosineDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
            mean_func_lr = self.warmup_factor * self.mean_func_lr
        # elif self._step_count >= self.tot_updates:
        #     lr = self.end_lr
        else:
            warmup = self.warmup_updates
            # lr_range = self.lr - self.end_lr
            # pct_remaining = 1 - (self._step_count - warmup) / (
            #     self.tot_updates - warmup
            # )
            # lr = lr_range * pct_remaining ** (self.power) + self.end_lr
            lr_step_range = self.tot_updates - warmup
            lf = lambda x: ((1 + math.cos(x * math.pi / lr_step_range)) / 2) * (1 - self.lrf) + self.lrf
            lr = lf(self._step_count - warmup) * self.lr
            mean_func_lr = lf(self._step_count - warmup) * self.mean_func_lr

        return [mean_func_lr for _ in range(self.view)] + [lr]
        # return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False