"""Bar-level prop-firm challenge simulator (numba).

Rules modelled (defaults = the user's challenge):
  * profit target      +10% of initial balance (equity, positions closed)
  * max daily loss      3% of initial, measured from the day's reference
                        (max of balance/equity at server midnight)
  * max overall loss    6% of initial (static floor at 94%)
  * time limit          14 calendar days
Execution: market entries at the next bar open, prices treated as mid with a
half-spread paid on every fill plus slippage on market/stop fills. Inside a
bar the adverse extreme is assumed to come first (stop before target), and
every open position is assumed to hit its adverse extreme simultaneously when
checking the equity limits, so results are conservative.

Trades come from strategies as arrays (sorted by entry bar):
  tk   instrument column          t0  entry bar (fill at open[t0])
  dirn +1 long / -1 short         stop  protective stop price
  tgt  take-profit price or nan   t1  last bar held (exit at close[t1])
"""
import numpy as np
from numba import njit

# parameter vector layout
P_RISK, P_HALT, P_DAYLIM, P_MAXLOSS, P_TARGET, P_MAXPOS, P_TBUF, P_LEV, \
    P_ROOMFRAC, P_TOTBUF, P_NEARTGT, P_ENDGAME_T, P_ENDGAME_MULT, P_RISK_EQ = range(14)
NPARAM = 14


def params(risk=0.01, halt=0.025, daylim=0.03, maxloss=0.06, target=0.10, maxpos=1,
           tbuf=0.0005, lev=30.0, roomfrac=1.0, totbuf=0.002, neartgt=0.0,
           endgame_t=2.0, endgame_mult=1.0, risk_eq=0.0):
    p = np.zeros(NPARAM)
    p[:] = (risk, halt, daylim, maxloss, target, maxpos, tbuf, lev, roomfrac, totbuf,
            neartgt, endgame_t, endgame_mult, risk_eq)
    return p


@njit(cache=True)
def _liq_equity(bal, npos, pk, pd, pq, pe, px, halfc):
    eq = bal
    for i in range(npos):
        k = pk[i]
        eq += pd[i] * pq[i] * (px[k] - pe[i]) - pq[i] * halfc[k]
    return eq


@njit(cache=True)
def run_challenge(s, e, O, H, L, C, day, halfc, slip,
                  tk, t0, dirn, stop, tgt, t1, ent, j0, p):
    """Simulate one challenge over bars s..e (inclusive).
    Returns (status, end_bar, final_equity_frac, n_trades, min_equity_frac)
    status: 1 pass, -1 fail, 0 timeout."""
    K = O.shape[1]
    init = 1.0
    bal = init
    maxpos = int(p[P_MAXPOS])
    pk = np.zeros(maxpos, np.int64)
    pd = np.zeros(maxpos)
    pq = np.zeros(maxpos)
    pe = np.zeros(maxpos)
    ps = np.zeros(maxpos)
    pt = np.zeros(maxpos)
    p1 = np.zeros(maxpos, np.int64)
    npos = 0
    last = np.empty(K)
    for k in range(K):
        last[k] = np.nan
    # find last valid closes before s for MTM
    for k in range(K):
        for b in range(s, max(s - 2000, -1), -1):
            if not np.isnan(C[b, k]):
                last[k] = C[b, k]
                break
    target_lvl = init * (1.0 + p[P_TARGET] + p[P_TBUF])
    total_floor = init * (1.0 - p[P_MAXLOSS])
    day_ref = init
    halted = False
    ntr = 0
    mineq = init
    j = j0
    nT = tk.shape[0]
    wc = np.empty(K)
    bc = np.empty(K)
    span = e - s + 1
    for t in range(s, e + 1):
        # ---------------- new day -----------------
        if t == s or day[t] != day[t - 1]:
            eq_prev = _liq_equity(bal, npos, pk, pd, pq, pe, last, halfc)
            day_ref = max(bal, eq_prev)
            halted = False
        halt_lvl = day_ref - p[P_HALT] * init
        day_floor = day_ref - p[P_DAYLIM] * init
        # ---------------- gap at open / stops at open --------------
        for k in range(K):
            if not np.isnan(O[t, k]):
                last[k] = O[t, k]
        i = 0
        while i < npos:
            k = pk[i]
            o = O[t, k]
            if not np.isnan(o) and ((pd[i] > 0 and o <= ps[i]) or (pd[i] < 0 and o >= ps[i])):
                fill = o - pd[i] * (halfc[k] + slip[k])
                bal += pd[i] * pq[i] * (fill - pe[i])
                npos -= 1
                pk[i] = pk[npos]; pd[i] = pd[npos]; pq[i] = pq[npos]; pe[i] = pe[npos]
                ps[i] = ps[npos]; pt[i] = pt[npos]; p1[i] = p1[npos]
            else:
                i += 1
        eq_open = _liq_equity(bal, npos, pk, pd, pq, pe, last, halfc)
        if eq_open <= halt_lvl and npos > 0:
            # EA closes everything at the open
            for i in range(npos):
                k = pk[i]
                bal += pd[i] * pq[i] * (last[k] - pd[i] * (halfc[k] + slip[k]) - pe[i])
            npos = 0
            halted = True
        if bal < day_floor or bal < total_floor:
            return -1, t, bal, ntr, min(mineq, bal)
        # ---------------- entries at open -----------------
        while j < nT and t0[j] < t:
            j += 1
        while j < nT and t0[j] == t:
            k = tk[j]
            if halted or np.isnan(O[t, k]):
                j += 1
                continue
            # existing position in same instrument?
            same = -1
            for i in range(npos):
                if pk[i] == k:
                    same = i
            if same >= 0:
                if pd[same] == dirn[j]:
                    j += 1
                    continue
                # reverse: close existing at open
                i = same
                fill = O[t, k] - pd[i] * (halfc[k] + slip[k])
                bal += pd[i] * pq[i] * (fill - pe[i])
                npos -= 1
                pk[i] = pk[npos]; pd[i] = pd[npos]; pq[i] = pq[npos]; pe[i] = pe[npos]
                ps[i] = ps[npos]; pt[i] = pt[npos]; p1[i] = p1[npos]
            if npos >= maxpos:
                j += 1
                continue
            d = dirn[j]
            px = O[t, k] if np.isnan(ent[j]) else ent[j]
            fill = px + d * (halfc[k] + slip[k])
            dist = abs(fill - stop[j]) + halfc[k] + slip[k]
            if (d > 0 and stop[j] >= fill) or (d < 0 and stop[j] <= fill) or dist <= 0:
                j += 1
                continue
            eq = _liq_equity(bal, npos, pk, pd, pq, pe, last, halfc)
            base = p[P_RISK] * (eq if p[P_RISK_EQ] > 0 else init)
            frac_el = (t - s) / span
            if frac_el >= p[P_ENDGAME_T]:
                base *= p[P_ENDGAME_MULT]
            open_risk = 0.0
            for i in range(npos):
                open_risk += pq[i] * (abs(pe[i] - ps[i]) + halfc[pk[i]] + slip[pk[i]])
            room_day = (eq - halt_lvl - open_risk) * p[P_ROOMFRAC]
            room_tot = (eq - total_floor - p[P_TOTBUF] * init - open_risk) * p[P_ROOMFRAC]
            risk_amt = min(base, room_day, room_tot)
            if p[P_NEARTGT] > 0 and not np.isnan(tgt[j]):
                gain_per = abs(tgt[j] - fill) - halfc[k]
                need = (target_lvl - eq) * p[P_NEARTGT]
                if gain_per > 0:
                    risk_amt = min(risk_amt, max(need, 0.0) / gain_per * dist)
            if risk_amt <= 1e-5 * init:
                j += 1
                continue
            q = risk_amt / dist
            qmax = p[P_LEV] * eq / fill
            if q > qmax:
                q = qmax
            pk[npos] = k; pd[npos] = d; pq[npos] = q; pe[npos] = fill
            ps[npos] = stop[j]; pt[npos] = tgt[j]; p1[npos] = t1[j]
            npos += 1
            ntr += 1
            j += 1
        # ---------------- intrabar: stops, then limits, then targets ------
        stopped_any = False
        i = 0
        while i < npos:
            k = pk[i]
            hi = H[t, k]
            lo = L[t, k]
            if np.isnan(hi):
                i += 1
                continue
            hit = (pd[i] > 0 and lo <= ps[i]) or (pd[i] < 0 and hi >= ps[i])
            if hit:
                fill = ps[i] - pd[i] * (halfc[k] + slip[k])
                bal += pd[i] * pq[i] * (fill - pe[i])
                npos -= 1
                pk[i] = pk[npos]; pd[i] = pd[npos]; pq[i] = pq[npos]; pe[i] = pe[npos]
                ps[i] = ps[npos]; pt[i] = pt[npos]; p1[i] = p1[npos]
                stopped_any = True
            else:
                i += 1
        # worst case equity
        for k in range(K):
            wc[k] = last[k]
            bc[k] = last[k]
        wce = bal
        bce = bal
        for i in range(npos):
            k = pk[i]
            if np.isnan(H[t, k]):
                adv = last[k]
                fav = last[k]
            elif pd[i] > 0:
                adv = L[t, k]
                fav = H[t, k]
            else:
                adv = H[t, k]
                fav = L[t, k]
            wce += pd[i] * pq[i] * (adv - pe[i]) - pq[i] * halfc[k]
            bce += pd[i] * pq[i] * (fav - pe[i]) - pq[i] * halfc[k]
        if npos > 0 and wce <= halt_lvl:
            # equity stop: flatten at the halt level, pay slippage
            sl = 0.0
            for i in range(npos):
                sl += pq[i] * slip[pk[i]]
            bal = halt_lvl - sl
            npos = 0
            halted = True
            wce = bal
        mineq = min(mineq, wce, bal)
        if bal < day_floor or bal < total_floor:
            return -1, t, bal, ntr, mineq
        if bal <= halt_lvl:
            halted = True
        # pass intrabar (no stop this bar, EA closes at target)
        if npos > 0 and not stopped_any and bce >= target_lvl:
            sl = 0.0
            for i in range(npos):
                sl += pq[i] * slip[pk[i]]
            return 1, t, target_lvl - sl, ntr, mineq
        # take profits
        i = 0
        while i < npos:
            k = pk[i]
            hi = H[t, k]
            lo = L[t, k]
            if np.isnan(hi) or np.isnan(pt[i]):
                i += 1
                continue
            hit = (pd[i] > 0 and hi >= pt[i]) or (pd[i] < 0 and lo <= pt[i])
            if hit:
                fill = pt[i] - pd[i] * halfc[k]
                bal += pd[i] * pq[i] * (fill - pe[i])
                npos -= 1
                pk[i] = pk[npos]; pd[i] = pd[npos]; pq[i] = pq[npos]; pe[i] = pe[npos]
                ps[i] = ps[npos]; pt[i] = pt[npos]; p1[i] = p1[npos]
            else:
                i += 1
        # ---------------- close of bar -----------------
        for k in range(K):
            if not np.isnan(C[t, k]):
                last[k] = C[t, k]
        i = 0
        while i < npos:
            if p1[i] <= t:
                k = pk[i]
                fill = last[k] - pd[i] * (halfc[k] + slip[k])
                bal += pd[i] * pq[i] * (fill - pe[i])
                npos -= 1
                pk[i] = pk[npos]; pd[i] = pd[npos]; pq[i] = pq[npos]; pe[i] = pe[npos]
                ps[i] = ps[npos]; pt[i] = pt[npos]; p1[i] = p1[npos]
            else:
                i += 1
        eq = _liq_equity(bal, npos, pk, pd, pq, pe, last, halfc)
        if eq >= target_lvl:
            sl = 0.0
            for i in range(npos):
                sl += pq[i] * slip[pk[i]]
            return 1, t, eq - sl, ntr, mineq
        if npos == 0 and bal >= target_lvl:
            return 1, t, bal, ntr, mineq
    eq = _liq_equity(bal, npos, pk, pd, pq, pe, last, halfc)
    return 0, e, eq, ntr, mineq


@njit(cache=True)
def run_all(starts, ends, O, H, L, C, day, halfc, slip, tk, t0, dirn, stop, tgt, t1, ent, p):
    n = starts.shape[0]
    out = np.zeros((n, 5))
    for c in range(n):
        s = starts[c]
        j0 = np.searchsorted(t0, s)
        st, eb, fe, nt, me = run_challenge(s, ends[c], O, H, L, C, day, halfc, slip,
                                           tk, t0, dirn, stop, tgt, t1, ent, j0, p)
        out[c, 0] = st
        out[c, 1] = eb
        out[c, 2] = fe
        out[c, 3] = nt
        out[c, 4] = me
    return out


@njit(cache=True)
def trade_R(O, H, L, C, halfc, slip, tk, t0, dirn, stop, tgt, t1, ent):
    """Outcome of every trade in isolation, in R (multiples of initial risk
    incl. costs). Returns (R, exit_bar)."""
    n = tk.shape[0]
    R = np.full(n, np.nan)
    xb = np.zeros(n, np.int64)
    T = O.shape[0]
    for j in range(n):
        k = tk[j]
        t = t0[j]
        if t >= T or np.isnan(O[t, k]):
            continue
        d = dirn[j]
        px = O[t, k] if np.isnan(ent[j]) else ent[j]
        fill = px + d * (halfc[k] + slip[k])
        dist = abs(fill - stop[j]) + halfc[k] + slip[k]
        if (d > 0 and stop[j] >= fill) or (d < 0 and stop[j] <= fill):
            continue
        ex = np.nan
        b = t
        lastc = px
        while b < T:
            o = O[b, k]
            if not np.isnan(o):
                if b > t and ((d > 0 and o <= stop[j]) or (d < 0 and o >= stop[j])):
                    ex = o - d * (halfc[k] + slip[k])
                    break
                hi = H[b, k]
                lo = L[b, k]
                if (d > 0 and lo <= stop[j]) or (d < 0 and hi >= stop[j]):
                    ex = stop[j] - d * (halfc[k] + slip[k])
                    break
                if not np.isnan(tgt[j]) and ((d > 0 and hi >= tgt[j]) or (d < 0 and lo <= tgt[j])):
                    ex = tgt[j] - d * halfc[k]
                    break
                lastc = C[b, k]
            if b >= t1[j]:
                ex = lastc - d * (halfc[k] + slip[k])
                break
            b += 1
        if np.isnan(ex):
            ex = lastc - d * (halfc[k] + slip[k])
        R[j] = d * (ex - fill) / dist
        xb[j] = b
    return R, xb


@njit(cache=True)
def resolve_pending(O, H, L, tk, t0, texp, lvl, dirn, grp):
    """Stop-entry orders: active from bar t0 through texp, fill when price
    trades through lvl (at the open if it gaps through). Orders sharing a
    grp >= 0 are OCO: the first to fill cancels the rest (same-bar ties go
    to the level nearer the bar open). Returns (fill_bar, fill_price)."""
    n = tk.shape[0]
    fb = np.full(n, -1, np.int64)
    fp = np.full(n, np.nan)
    for j in range(n):
        k = tk[j]
        for b in range(t0[j], texp[j] + 1):
            o = O[b, k]
            if np.isnan(o):
                continue
            if dirn[j] > 0 and H[b, k] >= lvl[j]:
                fb[j] = b
                fp[j] = max(o, lvl[j])
                break
            if dirn[j] < 0 and L[b, k] <= lvl[j]:
                fb[j] = b
                fp[j] = min(o, lvl[j])
                break
    # OCO
    order = np.argsort(grp)
    i = 0
    while i < n:
        g = grp[order[i]]
        e = i
        while e < n and grp[order[e]] == g:
            e += 1
        if g >= 0 and e - i > 1:
            best = -1
            for m in range(i, e):
                j = order[m]
                if fb[j] < 0:
                    continue
                if best < 0 or fb[j] < fb[best] or (fb[j] == fb[best] and
                        abs(fp[j] - O[fb[j], tk[j]]) < abs(fp[best] - O[fb[best], tk[best]])):
                    best = j
            for m in range(i, e):
                j = order[m]
                if j != best:
                    fb[j] = -1
                    fp[j] = np.nan
        i = e
    return fb, fp
