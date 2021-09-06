__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

from . import kt, models


class TAC:
    def __init__(self, tac, dt):
        self.tac = tac
        self.dt = dt
        self.mft = kt.dt2mft(dt)

        self.km_results = []

    def run_model(self, model_name, model_inputs):
        km_results = {'model_name': model_name}
        model = getattr(models, model_name)
        kps = model(self.tac, **model_inputs)
        km_results.update(kps)
        self.km_results = km_results

    def run_model_para2tac(self, model_name, model_inputs):
        km_results = {'model_name': model_name}
        model = getattr(models, model_name)
        kps = model(**model_inputs)
        km_results.update(kps)
        self.km_results = km_results


class Ref:
    def __init__(self, tac, dt):
        self.tac = tac
        self.dt = dt
        self.inputf1_feng_srtm = []
        self.inputf1 = []
        self.inputf1cubic = []
        self.inputf1_exp1 = []
        self.inputf1_exp2 = []
        self.inputf1_exp_am = []
        self.input_interp_method = []
        self.input_interp_1 = []

    def run_feng_srtm(self, w=None):
        self.inputf1_feng_srtm, _ = models.feng_srtm(self.tac, self.dt, w=w, fig=True)
        self.input_interp_1 = self.inputf1_feng_srtm

    def interp_1(self):
        mft = kt.dt2mft(self.dt)
        self.inputf1 = kt.interpt1(mft, self.tac, self.dt)
        self.input_interp_1 = self.inputf1

    def interp_1cubic(self):
        mft = kt.dt2mft(self.dt)
        self.inputf1cubic = kt.interpt1cubic(mft, self.tac, self.dt)
        self.input_interp_1 = self.inputf1cubic

    def run_exp1(self, w=None, idx_to_fit=None, fill_in_seconds=None):
        self.interp_1()
        self.inputf1_exp1 = self.inputf1
        inputf1_exp1 = 0
        if idx_to_fit is not None:
            inputf1_exp1, _ = models.exp_1(self.tac, self.dt, idx=idx_to_fit, w=w, fig=True)
        if fill_in_seconds is not None:
            self.inputf1_exp1[fill_in_seconds] = inputf1_exp1[fill_in_seconds]
        self.input_interp_1 = self.inputf1_exp1

    def run_exp2(self, w=None, idx_to_fit=None, fill_in_seconds=None):
        self.interp_1()
        self.inputf1_exp2 = self.inputf1
        inputf1_exp2 = 0
        if idx_to_fit is not None:
            inputf1_exp2, _ = models.exp_2(self.tac, self.dt, idx=idx_to_fit, w=w, fig=True)
        if fill_in_seconds is not None:
            self.inputf1_exp2[fill_in_seconds] = inputf1_exp2[fill_in_seconds]
        self.input_interp_1 = self.inputf1_exp2

    def run_exp_am(self, idx_to_fit=None, fill_in_seconds=None):
        self.interp_1()
        self.inputf1_exp_am = self.inputf1
        inputf1_exp_am = 0
        if idx_to_fit is not None:
            inputf1_exp_am, _ = models.exp_am(self.tac, self.dt, idx=idx_to_fit, fig=True)
        if fill_in_seconds is not None:
            self.inputf1_exp_am[fill_in_seconds] = inputf1_exp_am[fill_in_seconds]
        self.input_interp_1 = self.inputf1_exp_am

    def run_interp(self, input_interp_method='linear', **kwargs):
        input_interp_methods = {
            'linear': 'interp_1', 'cubic': 'interp_1cubic', 'exp_1': 'run_exp1',
            'exp_2': 'run_exp2', 'exp_am': 'run_exp_am', 'feng_srtm': 'run_feng_srtm'}
        self.input_interp_method = input_interp_method
        interp_method = getattr(self, input_interp_methods[input_interp_method])
        interp_method(self, **kwargs)
