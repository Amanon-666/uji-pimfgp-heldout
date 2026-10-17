"""跨楼层先验：由相邻旧楼层的实测指纹，给出目标楼层每个坐标上的先验地图。

目标层 AP a 在位置 p 的潜在水平 ℓ（dBm）。每条扫描：检出 ~ Bernoulli(h·Φ((ℓ−θ)/s))，检出值 ~ N(ℓ, σ_t²)。
旧层该位置可靠地检出过 AP 时（检出均值 m_L，k_L 次）：
  ℓ ~ N(CEN + ρ_z(m_L − CEN) + β_z, s_b²(z) + s_w² + ρ_z²σ_t²/k_L)
这是 Kennedy–O'Hagan 自回归迁移：ρ_z 为域相关系数，β_z 为整体偏移，s_b² 为逐 AP 偏移方差，s_w² 为单元级偏差。
z 为 AP 在旧层的最强可靠检出档，代理 AP 相对两层的位置。旧层从未检出的单元是删失：ℓ ~ N(μ0(z), s_b² + s_w² + v_L0(z))。
全部参数只在源楼层对（上下层同坐标单元）上估计：方差分量用矩估计；β、ρ、θ、s、μ0、v_L0 用删失复合似然
（检出次数与检出值共用同一个潜变量）。
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr

NO_SIGNAL = -110.0
CEN = -75.0
EDGES = np.array([-80.0, -70.0, -60.0, -50.0])
NB = len(EDGES) + 2
GH_T, GH_W = np.polynomial.hermite.hermgauss(32)
LOG_GH_W = np.log(GH_W / np.sqrt(np.pi))


def ap_bin(lc):
    """AP 档：0 为旧层从未可靠检出（q≥0.5），1–5 按旧层最强可靠检出均值 (<−80, −80…−70, …, >−50 dBm)。"""
    q = lc["k"] / lc["n"][:, None]
    m = np.where(q >= 0.5, lc["s1"] / np.maximum(lc["k"], 1), -np.inf).max(0)
    return np.where(np.isfinite(m), 1 + np.searchsorted(EDGES, m), 0)


def nearest(xy, ref):
    d = np.sqrt(((xy[:, None] - ref[None]) ** 2).sum(-1))
    return d.argmin(1), d.min(1)


def cell_loglik(mu, v, n, k, m, st, h, theta, s):
    """单元边缘对数似然 log ∫ N(ℓ; μ, v)·Binom(k; n, h·Φ((ℓ−θ)/s))·Π_j N(x_j; ℓ, σ_t²) dℓ（略去与参数无关的项）。
    检出值项对 ℓ 是高斯，先解析合并；检出次数项用 Gauss–Hermite 积分。"""
    kk = np.maximum(k, 1)
    has = k > 0
    vc = np.where(has, 1 / (1 / v + kk / st), v)
    mc = np.where(has, vc * (mu / v + kk * m / st), mu)
    val = np.where(has, -0.5 * (m - mu) ** 2 / (v + st / kk) - 0.5 * np.log(v + st / kk), 0.0)
    L = mc[..., None] + np.sqrt(2 * vc)[..., None] * GH_T
    g = np.clip(h * ndtr((L - theta) / s), 1e-12, 1 - 1e-12)
    lp = k[..., None] * np.log(g) + (n - k)[..., None] * np.log1p(-g) + LOG_GH_W
    return val + logsumexp(lp, axis=-1)


def _pair_arrays(pair_cells):
    """上下层共位单元：BOTH 两层都检出（矩估计）、LINK 旧层检出（复合似然）、UND 旧层删失（按档）。"""
    BOTH, G, LINK = [], [], []
    UND = [[] for _ in range(NB)]
    for t, (lc, uc) in enumerate(pair_cells):
        z = ap_bin(lc)
        l, dist = nearest(uc["xy"], lc["xy"])
        u = np.flatnonzero(dist <= 1.0)
        l = l[u]
        kL, kU = lc["k"][l], uc["k"][u]
        nU = np.repeat(uc["n"][u][:, None], kL.shape[1], 1)
        mL, mU = lc["s1"][l] / np.maximum(kL, 1), uc["s1"][u] / np.maximum(kU, 1)
        uni = (lc["k"].sum(0) > 0) | (kU.sum(0) > 0)
        zz = np.broadcast_to(z, kL.shape)
        det = (kL >= 1) & uni
        i, a = np.nonzero(det & (kU >= 1))
        BOTH.append(np.stack([mU[i, a], mL[i, a], kU[i, a], kL[i, a], z[a]], 1))
        G.append(a + 1000 * t)
        LINK.append(np.stack([mL[det], kL[det], zz[det], kU[det], nU[det], mU[det]], 1))
        for b in range(NB):
            sel = (kL == 0) & uni & (zz == b)
            UND[b].append(np.stack([kU[sel], nU[sel], mU[sel]], 1))
    return np.concatenate(BOTH), np.concatenate(G), np.concatenate(LINK), UND


def _link_loglik(tf, st, link):
    lm, lk, lz, lku, lnu, lmu = link.T
    lz = lz.astype(int)
    r = tf["rho"][lz]
    mu = CEN + r * (lm - CEN) + tf["bbar"][lz]
    v = tf["sb2"][lz] + tf["s_w2"] + r ** 2 * st / lk
    return cell_loglik(mu, v, lnu, lku, lmu, st, tf["h"], tf["theta"], tf["s"])


def fit_transfer(pair_cells, obs, sweeps=3):
    """源楼层对上估计跨楼层先验。方差分量（给定 ρ）与 (β, ρ, θ, s)（给定方差分量）交替估计 sweeps 轮。"""
    st, h = obs["sigma_total"] ** 2, obs["h_det"]
    BOTH, G, LINK, UND = _pair_arrays(pair_cells)
    _, gi, gn = np.unique(G, return_inverse=True, return_counts=True)
    Zb = BOTH[:, 4].astype(int)
    gz = np.round(np.bincount(gi, Zb) / gn).astype(int)
    big, ok = gn >= 3, gn[gi] >= 3

    def components(rho):
        r = rho[Zb]
        R = BOTH[:, 0] - CEN - r * (BOTH[:, 1] - CEN)
        N = st * (1 / BOTH[:, 2] + r ** 2 / BOTH[:, 3])
        bhat = np.bincount(gi, R) / gn
        e = R - bhat[gi]
        s_w2 = max(float(np.mean(e[ok] ** 2 * gn[gi][ok] / (gn[gi][ok] - 1) - N[ok])), 1.0)
        gnoise = np.bincount(gi, N) / gn
        b0, sb2 = np.zeros(NB), np.zeros(NB)
        for b in range(NB):
            sel = big & (gz == b)
            sel = sel if sel.sum() >= 5 else big
            b0[b] = bhat[sel].mean()
            sb2[b] = max(bhat[sel].var() - np.mean((s_w2 + gnoise[sel]) / gn[sel]), 1.0)
        return s_w2, b0, sb2

    rho, x = np.ones(NB), None
    for _ in range(sweeps):
        s_w2, b0, sb2 = components(rho)
        cur = {"sb2": sb2, "s_w2": s_w2, "h": h}

        def nll(p):
            cur.update(bbar=p[:NB], rho=p[NB:2 * NB], theta=p[2 * NB], s=np.exp(p[2 * NB + 1]))
            return -_link_loglik(cur, st, LINK).mean()

        start = np.r_[b0, rho, -85.0, np.log(12.0)] if x is None else x
        bounds = [(None, None)] * NB + [(0.0, 1.5)] * NB + [(None, None), (np.log(0.5), np.log(60.0))]
        x = minimize(nll, start, method="L-BFGS-B", bounds=bounds).x
        rho = x[NB:2 * NB].copy()
    bbar, theta, s = x[:NB], float(x[2 * NB]), float(np.exp(x[2 * NB + 1]))
    link = float(nll(x))
    # 旧层删失单元：每档一个先验 N(μ0, s_b² + s_w² + v_L0)，同一似然；k=0 的单元按 n 聚合
    mu0, vL0 = np.zeros(NB), np.zeros(NB)
    pooled = np.concatenate([np.concatenate(u) for u in UND])
    for b in range(NB):
        arr = np.concatenate(UND[b])
        arr = arr if arr[:, 1].sum() >= 200 else pooled
        hit = arr[arr[:, 0] >= 1]
        n0, c0 = np.unique(arr[arr[:, 0] == 0, 1], return_counts=True)
        kk, nn = np.r_[hit[:, 0], np.zeros(len(n0))], np.r_[hit[:, 1], n0]
        mm, ww = np.r_[hit[:, 2], np.zeros(len(n0))], np.r_[np.ones(len(hit)), c0]
        one, shared = np.ones(len(kk)), sb2[b] + s_w2
        nll_c = lambda y: -(ww * cell_loglik(y[0] * one, (shared + np.exp(y[1])) * one, nn, kk, mm, st, h, theta, s)).sum() / ww.sum()
        y = minimize(nll_c, [-100.0, np.log(100.0)], method="L-BFGS-B").x
        mu0[b], vL0[b] = y[0], float(np.exp(y[1]))
    return {"bbar": bbar, "rho": rho, "sb2": sb2, "s_w2": s_w2, "h": h, "theta": theta, "s": s,
            "mu0": mu0, "vL0": vL0, "sigma_total": obs["sigma_total"], "link_nll": link,
            "n_cells": int(len(BOTH)), "n_link_cells": int(len(LINK)), "n_groups": int(big.sum())}


def prior_map(lc, tf):
    """零标注先验地图：旧层每个坐标上，目标层观测的期望插补值（未检出记 −110 dBm）
    E[x_imp] = h(Φ(z)μ + vφ(z)/√(s²+v)) + (1 − hΦ(z))·(−110)，z = (μ−θ)/√(s²+v)。旧层从未检出的 AP 记 −110。"""
    st = tf["sigma_total"] ** 2
    kL = lc["k"]
    det = kL >= 1
    zb = np.broadcast_to(ap_bin(lc), kL.shape)
    rho = tf["rho"][zb]
    mu = np.where(det, CEN + rho * (lc["s1"] / np.maximum(kL, 1) - CEN) + tf["bbar"][zb], tf["mu0"][zb])
    v = tf["sb2"][zb] + tf["s_w2"] + np.where(det, rho ** 2 * st / np.maximum(kL, 1), tf["vL0"][zb])
    sd = np.sqrt(tf["s"] ** 2 + v)
    z = (mu - tf["theta"]) / sd
    p = tf["h"] * ndtr(z)
    x = tf["h"] * (ndtr(z) * mu + v * np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sd)) + (1 - p) * NO_SIGNAL
    x[:, lc["k"].sum(0) == 0] = NO_SIGNAL
    return np.clip(x, NO_SIGNAL, 0), lc["xy"]
