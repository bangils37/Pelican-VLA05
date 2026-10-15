"""Attention graph propagation. NumPy only; no model/GPU dependency.

Rollout: Abnar & Zuidema (2020), residual addition and row normalization.
Chefer aggregation: positive gradient * LRP relevance, head mean, residual
addition without row normalization. The caller MUST supply actual LRP relevance.
"""
import numpy as np


def chefer_weights(relevance, gradients):
    """CVPR 2021 attention weighting, inputs [layers, heads, queries, keys].

    This is the aggregation stage only, NOT a substitute for model relprop.
    """
    r, g = np.asarray(relevance), np.asarray(gradients)
    if r.shape != g.shape or r.ndim != 4:
        raise ValueError("relevance and gradients must have matching [L,H,Q,K] shapes")
    if not np.isfinite(r).all() or not np.isfinite(g).all():
        raise ValueError("non-finite relevance or gradients")
    return np.maximum(r * g, 0).mean(axis=1)


def cached_rollout(prefix, continuation, suffix, image_columns, *, slots=0,
                   normalize=True):
    """Propagate image-token provenance through cached decoder layers.

    Inputs are head-fused [L,Q,K]. Continuation keys: prefix + continuation.
    Suffix keys: last `slots` continuation tokens (if enabled), otherwise
    prefix + continuation; followed by suffix tokens. Each layer reads the
    PREVIOUS layer's provenance, including cached K/V. Returns [suffix,Qimage].
    Cosmos/middle, state and language are not image-token seeds; image mass
    need not sum to one. This is a language-decoder graph attribution only.
    """
    p, c, s = [np.asarray(a, dtype=np.float32) for a in (prefix, continuation, suffix)]
    if any(a.ndim != 3 or not np.isfinite(a).all() or (a < 0).any() for a in (p,c,s)):
        raise ValueError("expected finite nonnegative [layers,queries,keys] weights")
    L, P, _ = p.shape
    C, S = c.shape[1], s.shape[1]
    memory = slots if slots else P + C
    if (L == 0 or p.shape != (L,P,P) or c.shape != (L,C,P+C)
            or s.shape != (L,S,memory+S) or not 0 <= slots <= C):
        raise ValueError("attention shapes do not match Pelican cache layout")
    cols = np.asarray(image_columns, dtype=np.int64)
    if cols.ndim != 1 or len(cols) == 0 or (cols < 0).any() or (cols >= P).any():
        raise ValueError("invalid image columns")
    rp = np.zeros((P,len(cols)), dtype=np.float32)
    rp[cols, np.arange(len(cols))] = 1
    rc = np.zeros((C,len(cols)), dtype=np.float32)
    rs = np.zeros((S,len(cols)), dtype=np.float32)
    def update(a, keys, residual):
        out = a @ keys + residual
        return out / (a.sum(-1, keepdims=True) + 1) if normalize else out
    for layer in range(L):
        pc = np.concatenate((rp, rc))
        mem = rc[-slots:] if slots else pc
        new_s = update(s[layer], np.concatenate((mem,rs)), rs)
        new_c = update(c[layer], pc, rc)
        rp = update(p[layer], rp, rp)
        rc, rs = new_c, new_s
    return rs
