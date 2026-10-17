# mock 桩。注意：mock 的 AP 筛选已对齐为"丢弃在所有候选上恒为 NO_SIGNAL 的 AP"（与 mapsep.py 的 G3 全 AP 打分在数学上等价），
# 只为本地跑通 G3==law_other 硬断言。真实服务器上该断言必须对真实 tobit_match 成立，mock 对齐不代表真实实现。
import numpy as np
from scipy.special import log_ndtr
from pimfgp_heldout import dbm, NO_SIGNAL
def tobit_match(qx,P,xy,sig,th):
    det=qx!=100; x=dbm(qx); A=np.flatnonzero((P>NO_SIGNAL).any(0)); det=det[:,A]; x=x[:,A]; P=P[:,A]
    xd=x*det; d2=(xd*xd).sum(1,keepdims=True)+det.astype(float)@(P**2).T-2*xd@P.T
    S=-0.5/sig**2*d2+(~det).astype(float)@log_ndtr((th-P)/sig).T
    S-=S.max(1,keepdims=True); W=np.exp(S); W/=W.sum(1,keepdims=True); return W@xy
