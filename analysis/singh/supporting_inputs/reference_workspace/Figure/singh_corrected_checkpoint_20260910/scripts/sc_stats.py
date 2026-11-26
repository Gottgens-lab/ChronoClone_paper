"""Divergence statistics on the corrected vocabulary (item A1/A2 recompute).

This is `divergence_timing_20260910/scripts/dt_core.py` with the vocabulary repaired and
nothing else changed. Every test setting is held identical to the previous run
(n_perm=2000, min_cells=10, min_clones=3, seed=0, stratum = 10x sample
(replicate excluded from the design by data-owner instruction; see vocab.py),
stratum-averaged statistics weighted by min(n_wt, n_mut)) so that any difference in the
before/after table is attributable to the vocabulary repair alone and not to a changed
procedure. Do not tune these while the change table is the deliverable.

Two behavioural differences follow necessarily from the repair rather than from choice:

  * Compositions come from `vocab.composition`, which refuses to emit an all-zero row.
    Previously 78 clone x stratum rows were all-zero and became NaN under row
    normalisation; the stratum mean then skipped them per column, so different cell-type
    categories were averaged over different numbers of clones. Those rows now carry
    their real basophil-dominated composition.
  * Clone-level rows are restricted to resolved lineage identifiers, so the stringified
    NaN family no longer contributes a 1,517th clone.

`presence` keeps the previous, deliberate convention: a metaclone with zero cells in one
arm is an observed divergence, not a missing value, provided both arms have enough cells
in the stratum to have detected it. That is a statement about relative representation
among recovered cells and nothing more -- not expansion, not survival, not extinction.
"""
import numpy as np
import pandas as pd
from vocab import CELLTYPES, MCS, MUTANTS, composition


def clone_arm_table(D):
    """One row per resolved clone x day x arm x stratum: n, mean pseudotime, shares."""
    key = ["mc", "day", "arm", "stratum", "clone"]
    R = D[D.clone_resolved]
    g = R.groupby(key, observed=True)
    out = g.agg(n=("psi", "size"), psi=("psi", "mean")).reset_index()
    comp, ncell = composition(R, key)
    ca = out.merge(comp.reset_index(), on=key, how="left")
    assert ca[CELLTYPES].notna().all().all(), "a clone row has no composition"
    assert np.allclose(ca[CELLTYPES].sum(1), 1), "clone composition rows do not sum to 1"
    return ca


# ---------------------------------------------------------------- presence axis
def presence(D, min_arm_cells=20):
    rows = []
    R = D[D.clone_resolved]
    tot_cell = R.groupby(["stratum", "arm"], observed=True).size()
    n_cell = R.groupby(["mc", "stratum", "arm"], observed=True).size()
    n_clone = R.groupby(["mc", "stratum", "arm"], observed=True).clone.nunique()
    for mc in MCS:
        for day in ["D14", "D27"]:
            strata = sorted(R.loc[R.day == day, "stratum"].unique())
            for mut in MUTANTS:
                per = []
                for st in strata:
                    tw, tm = tot_cell.get((st, "WT"), 0), tot_cell.get((st, mut), 0)
                    if tw < min_arm_cells or tm < min_arm_cells:
                        continue
                    cw, cm = n_cell.get((mc, st, "WT"), 0), n_cell.get((mc, st, mut), 0)
                    kw, km = n_clone.get((mc, st, "WT"), 0), n_clone.get((mc, st, mut), 0)
                    sw, sm = cw / tw, cm / tm
                    eps = 0.5 / max(tw, tm)
                    per.append(dict(stratum=st, cw=cw, cm=cm, kw=kw, km=km,
                                    share_wt=sw, share_mut=sm,
                                    l2=np.log2((sm + eps) / (sw + eps))))
                if not per:
                    continue
                p = pd.DataFrame(per)
                rows.append(dict(mc=mc, day=day, mutant=mut, n_strata=len(p),
                                 cells_wt=int(p.cw.sum()), cells_mut=int(p.cm.sum()),
                                 clones_wt=int(p.kw.max()), clones_mut=int(p.km.max()),
                                 presence_l2=float(p.l2.mean()),
                                 presence_l2_min=float(p.l2.min()),
                                 presence_l2_max=float(p.l2.max()),
                                 wt_absent=bool(p.cw.sum() == 0),
                                 mut_absent=bool(p.cm.sum() == 0),
                                 per_stratum=";".join(f"{r.stratum}:{r.l2:+.2f}"
                                                      for r in p.itertuples())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- state axis
def _strat_stat(ca, mut, stat):
    vals, ws = [], []
    for st, sub in ca.groupby("stratum", observed=True):
        w, m = sub[sub.arm == "WT"], sub[sub.arm == mut]
        if len(w) < 2 or len(m) < 2:
            continue
        if stat == "dpsi":
            v = m.psi.mean() - w.psi.mean()
        else:
            pw, pm = w[CELLTYPES].mean().values, m[CELLTYPES].mean().values
            assert np.isfinite(pw).all() and np.isfinite(pm).all(), \
                "non-finite mean composition; the NaN-skipping defect has returned"
            v = 0.5 * np.abs(pm - pw).sum()
        vals.append(v)
        ws.append(min(len(w), len(m)))
    if not vals:
        return np.nan, 0
    return float(np.average(vals, weights=ws)), len(vals)


def _fast_stats(codes, ismut, psi, comp, nstr):
    key = codes * 2 + ismut
    cnt = np.bincount(key, minlength=2 * nstr).astype(float)
    spsi = np.bincount(key, weights=psi, minlength=2 * nstr)
    cw, cm = cnt[0::2], cnt[1::2]
    ok = (cw >= 2) & (cm >= 2)
    if not ok.any():
        return np.nan, np.nan
    mw = np.divide(spsi[0::2], cw, out=np.zeros(nstr), where=cw > 0)
    mm = np.divide(spsi[1::2], cm, out=np.zeros(nstr), where=cm > 0)
    w = np.minimum(cw, cm)[ok]
    d = float(np.average((mm - mw)[ok], weights=w))
    tv = np.zeros(nstr)
    for j in range(comp.shape[1]):
        s = np.bincount(key, weights=comp[:, j], minlength=2 * nstr)
        pw = np.divide(s[0::2], cw, out=np.zeros(nstr), where=cw > 0)
        pm = np.divide(s[1::2], cm, out=np.zeros(nstr), where=cm > 0)
        tv += np.abs(pm - pw)
    return d, float(np.average(0.5 * tv[ok], weights=w))


def state(ca, n_perm=2000, min_cells=10, min_clones=3, seed=0):
    """Within-stratum permutation of the arm label on clone rows.

    Untestable cells carry `testable=False` and NaN statistics with a stated reason.
    They are never zero and never p=1: not estimable is not evidence of no effect.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mc in MCS:
        for day in ["D14", "D27"]:
            for mut in MUTANTS:
                sub = ca[(ca.mc == mc) & (ca.day == day) &
                         (ca.arm.isin(["WT", mut]))].copy()
                nw = sub.loc[sub.arm == "WT", "n"].sum()
                nm = sub.loc[sub.arm == mut, "n"].sum()
                kw = int((sub.arm == "WT").sum())
                km = int((sub.arm == mut).sum())
                rec = dict(mc=mc, day=day, mutant=mut, cells_wt=int(nw), cells_mut=int(nm),
                           clones_wt=kw, clones_mut=km)
                if nw < min_cells or nm < min_cells or kw < min_clones or km < min_clones:
                    rec.update(testable=False,
                               reason=("WT not detected" if nw == 0 else
                                       "mutant not detected" if nm == 0 else
                                       "too few cells/clones"),
                               dpsi=np.nan, tvd=np.nan, p_dpsi=np.nan, p_tvd=np.nan,
                               n_strata=0)
                    rows.append(rec); continue
                d_obs, ns = _strat_stat(sub, mut, "dpsi")
                t_obs, _ = _strat_stat(sub, mut, "tvd")
                if not np.isfinite(d_obs):
                    rec.update(testable=False, reason="no stratum with both arms",
                               dpsi=np.nan, tvd=np.nan, p_dpsi=np.nan, p_tvd=np.nan,
                               n_strata=0)
                    rows.append(rec); continue
                strata = sub.stratum.unique()
                codes = pd.Categorical(sub.stratum, categories=strata).codes.astype(np.int64)
                ismut = (sub.arm.values == mut).astype(np.int64)
                psi = sub.psi.values.astype(float)
                comp = sub[CELLTYPES].values.astype(float)
                nstr = len(strata)
                idx_by_st = [np.where(codes == s)[0] for s in range(nstr)]
                dn, tn = np.empty(n_perm), np.empty(n_perm)
                for i in range(n_perm):
                    perm = ismut.copy()
                    for ix in idx_by_st:
                        perm[ix] = rng.permutation(ismut[ix])
                    dn[i], tn[i] = _fast_stats(codes, perm, psi, comp, nstr)
                dn, tn = dn[np.isfinite(dn)], tn[np.isfinite(tn)]
                rec.update(testable=True, reason="", n_strata=ns, dpsi=d_obs, tvd=t_obs,
                           p_dpsi=(1 + (np.abs(dn) >= abs(d_obs)).sum()) / (1 + len(dn)),
                           p_tvd=(1 + (tn >= t_obs).sum()) / (1 + len(tn)),
                           sd_null_dpsi=float(np.std(dn)), sd_null_tvd=float(np.std(tn)),
                           mde_dpsi=float(2.80 * np.std(dn)))
                rows.append(rec)
    return pd.DataFrame(rows)


def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]; order = np.argsort(pv); n = len(pv)
    adj = pv[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def paired_families(D, mut, day, mc=None, min_cells=2):
    """Same resolved lineage family, mutant vs WT, same day and same stratum."""
    sub = D[(D.day == day) & (D.arm.isin(["WT", mut])) & D.clone_resolved]
    if mc is not None:
        sub = sub[sub.mc == mc]
    g = sub.groupby(["clone", "stratum", "arm"], observed=True).agg(
        n=("psi", "size"), psi=("psi", "mean")).reset_index()
    w = g[g.arm == "WT"].set_index(["clone", "stratum"])
    m = g[g.arm == mut].set_index(["clone", "stratum"])
    j = w.drop(columns=["arm"]).join(m.drop(columns=["arm"]),
                                     lsuffix="_wt", rsuffix="_mut", how="inner")
    j = j[(j.n_wt >= min_cells) & (j.n_mut >= min_cells)]
    j["dpsi"] = j.psi_mut - j.psi_wt
    return j.reset_index()
