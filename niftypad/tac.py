__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

from . import models
from . import kt


class TAC:

    def __init__(self, tac, dt):
        self.tac = tac
        self.dt = dt
        self.mft = kt.dt2mft(dt)

        self.model = []
        self.r1 = []
        self.k2 = []
        self.bp = []
        self.tacf = []

    def run_srtm(self, inputf1, w):
        self.model = 'srtm'
        r1, k2, bp, tacf = models.srtm(self.tac, self.dt, inputf1, w)
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = tacf

    def run_srtm_k2p(self, inputf1, w, k2p):
        self.model = 'srtm_k2p'
        r1, k2, bp, tacf = models.srtm_k2p(self.tac, self.dt, inputf1, w, k2p)
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = tacf

    def run_srtmb(self, inputf1, w):
        self.model = 'srtmb'
        r1, k2, bp, tacf = models.srtmb(self.tac, self.dt, inputf1, w)
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = tacf

    def run_srtmb_basis(self, b):
        self.model = 'srtmb'
        r1, k2, bp, tacf = models.srtmb_basis(self.tac, b)
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = tacf

    def run_srtmb_k2p(self, inputf1, w, k2p):
        self.model = 'srtmb_k2p'
        r1, k2, bp, tacf = models.srtmb_k2p(self.tac, self.dt, inputf1, w, k2p)
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = tacf

    def run_srtmb_k2p_basis(self, b):
        self.model = 'srtmb_k2p'
        r1, k2, bp, tacf = models.srtmb_k2p_basis(self.tac, b)
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = tacf

    def run_srtmb_asl(self, inputf1, w, r1):
        self.model = 'srtmb_asl'
        r1, k2, bp, tacf = models.srtmb_asl(self.tac, self.dt, inputf1, w, r1)
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = tacf

    def run_logan_ref_k2p(self, inputf1, k2p, linear_phase_start, linear_phase_end):
        self.model = 'logan_ref_k2p'
        bp = models.logan_ref_k2p(self.tac, self.dt, inputf1, k2p, linear_phase_start, linear_phase_end)
        self.bp = bp

    def run_srtmb_basis_para2tac(self, r1, k2, bp, b):
        self.model = 'srtmb'
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = models.srtmb_basis_para2tac(r1, k2, bp, b)

    def run_srtm_para2tac(self, r1, k2, bp, inputf1_dt):
        self.model = 'srtm'
        self.r1 = r1
        self.k2 = k2
        self.bp = bp
        self.tacf = models.srtm_para2tac(r1, k2, bp, inputf1_dt)




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