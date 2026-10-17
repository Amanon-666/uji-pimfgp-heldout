import numpy as np
from scipy.special import log_ndtr
from pimfgp_heldout import dbm
def tobit_match(qx,P,xy,sig,th):
    det=qx!=100; x=dbm(qx); A=np.flatnonzero((P>-105).any(0)); det=det[:,A]; x=x[:,A]; P=P[:,A]
    xd=x*det; d2=(xd*xd).sum(1,keepdims=True)+det.astype(float)@(P**2).T-2*xd@P.T
    S=-0.5/sig**2*d2+(~det).astype(float)@log_ndtr((th-P)/sig).T
    S-=S.max(1,keepdims=True); W=np.exp(S); W/=W.sum(1,keepdims=True); return W@xy
