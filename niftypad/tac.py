__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

from . import models
from . import kt


class TAC:

    def __init__(self, tac, dt):
        self.tac = tac
        self.dt = dt
        self.mft = kt.dt2mft(dt)

        self.km_results = []

    def run_model(self, model_name, model_inputs):
        km_results = dict()
        km_results.update({'model_name': model_name})
        model = getattr(models, model_name)
        kps = model(self.tac, **model_inputs)
        km_results.update(kps)
        self.km_results.append(km_results)


class Ref:

    def __init__(self, tac, dt):
        self.tac = tac
        self.dt = dt
        self.inputf1_fs = []
        self.inputf1 = []
        self.inputf1cubic = []

    def run_feng_srtm(self, w=None):
        self.inputf1_fs, _ = models.feng_srtm(self.tac, self.dt, w=w, fig=True)

    def interp_1(self):
        mft = kt.dt2mft(self.dt)
        self.inputf1 = kt.interpt1(mft, self.tac, self.dt)

    def interp_1cubic(self):
        mft = kt.dt2mft(self.dt)
        self.inputf1cubic = kt.interpt1cubic(mft, self.tac, self.dt)