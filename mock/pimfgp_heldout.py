# 本地 mock：仅用于在盒子上检查 mapsep.py 逻辑能跑通，不代表服务器实现
import json, csv
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
D='/workspace/ujiloc/UJIndoorLoc/'
TRAIN=Path(D+'trainingData.csv'); VALID=Path(D+'validationData.csv'); SPLITS=Path(__file__).parent/'splits.json'
BUILDINGS={0:4,1:4,2:5}; NO_SIGNAL=-110.0; CFG={}; MATCHES=[(6.0,-95.0),(10.0,-95.0)]; SMOKE=True; D0=1.0
def dbm(r): r=np.asarray(r,float); return np.where(r==100,NO_SIGNAL,r)
def load_floors(p):
    X=np.loadtxt(p,delimiter=',',skiprows=1)
    out={}
    for b,nf in BUILDINGS.items():
        for f in range(nf):
            m=np.flatnonzero((X[:,522]==f)&(X[:,523]==b))
            out[f'B{b}F{f}']={'rssi':X[m,:520],'xy':X[m,520:522]-X[:,520:522].min(0),'row_ids':m}
    return out
def position_map(sx,sy):
    u,inv=np.unique(sy,axis=0,return_inverse=True); inv=inv.ravel()
    x=dbm(sx); P=np.stack([x[inv==i].mean(0) for i in range(len(u))]); return P,u,inv
def cells(fl): P,u,_=position_map(fl['rssi'],fl['xy']); return {'p':P,'xy':u}
def pairs_of(bs): return [(f'B{b}F{f}',f'B{b}F{f+1}') for b in bs for f in range(BUILDINGS[b]-1)]
def fit_law(Cc,pairs): return None,(1.0,-3.0)
def prior_map(c,law): return {'p':np.clip(law[0]*c['p']+law[1]*(c['p']>NO_SIGNAL),NO_SIGNAL,0),'xy':c['xy']}
def build_map(kind,base,sx,sy,field,cfg):
    P,u,_=position_map(sx,sy); off=(P-lookup_(u,base['xy'],base['p'],1)).mean(0)*0.5
    return (np.clip(base['p']+off,NO_SIGNAL,0),base['xy'])
def lookup_(q,xy,P,k):
    i=np.argsort(cdist(q,xy),1)[:,:k]; return P[i].mean(1)
def support_grid(sy,step=4.0,pad=15.0):
    lo=sy.min(0)-pad; hi=sy.max(0)+pad; gx=np.arange(lo[0],hi[0]+1e-9,step); gy=np.arange(lo[1],hi[1]+1e-9,step)
    return np.array([(a,b) for a in gx for b in gy])
def kde_logprior(G,sy,bw=10.0):
    return np.log(np.exp(-cdist(G,sy,'sqeuclidean')/(2*bw*bw)).mean(1)+1e-12)
def episodes(xy,n,E,s,rng):
    u,inv=np.unique(xy,axis=0,return_inverse=True); inv=inv.ravel()
    for _ in range(E):
        pick=rng.choice(len(u),n,replace=False); sm=np.isin(inv,pick)
        yield np.flatnonzero(sm)[:n*s], np.flatnonzero(~sm)
def supports(sp):
    for d in range(2):
        for k,ids in sp['supports'].items(): yield d,k,ids[d]
def targets_of(b): return [f'B{b}F{f}' for f in range(1,BUILDINGS[b])]
def rss_x(r): return dbm(r)
def wknn(sx,sy,qx,k=3):
    i=np.argsort(cdist(qx,sx),1)[:,:k]; return sy[i].mean(1)
def est_nbar(x): return 2.0
