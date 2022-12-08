import numpy as np


def teps_to_wndwsz(szr_len, t_eps, npoints):
    dt = szr_len / npoints
    wndw_sz = int(np.round(t_eps / dt))
    return wndw_sz


def find_bst_szr(pat_id, szr_names, data_dir):
    max_var = 0
    for name in szr_names:
        slp = np.load(f'{data_dir}/{pat_id}/fit/data_{name}.npz')['slp_obs']
        slp -= slp.mean(axis=0)
        var = (slp**2).mean(axis=0).var(axis=0)
        if var > max_var:
            max_var = var
            bst_szr = name
    return bst_szr


def find_best_map_run(res_dir, pat_id, szr_name, num_runs):
    losses = []
    for i in range(1, num_runs + 1):
        map_estim = np.load(
            f'{res_dir}/{pat_id}/map_estimate_{szr_name}_run{i:d}.npz')
        l = map_estim['losses'][-1, 0]
        if np.isnan(l):
            l = np.inf
        losses.append(l)
    best_map_run = np.argmin(losses) + 1
    return best_map_run, losses[best_map_run - 1]


def find_onsets(ts, thrshld):
    nt, nn = ts.shape
    onsets = (nt + 50) * np.ones(nn)
    for i in range(nn):
        ts_i_thrshd = ts[:, i] > thrshld
        if (ts_i_thrshd.any()):
            onsets[i] = np.nonzero(ts_i_thrshd)[0][0]
        else:
            onsets[i] = np.inf
    return onsets


def find_ez(x, src_thrshld, onst_wndw_sz):
    nt, nn = x.shape
    onsets = find_onsets(x, src_thrshld)
    nszng = np.size(np.nonzero(onsets < nt))
    assert nszng > 0, "No seizing regions found"
    first_onset_time = onsets.min()
    ez_pred = np.zeros(nn)
    onst_thrshld = first_onset_time + onst_wndw_sz
    ez_pred[np.nonzero(onsets <= onst_thrshld)[0]] = 1
    pz_pred = np.zeros(nn)
    pz_pred[np.nonzero(np.logical_and(onsets > onst_thrshld,
                                      onsets < nt))[0]] = 1
    return ez_pred, pz_pred


def find_ez_x0(x0, thrshld):
    ez_pred = np.zeros_like(x0)
    ez_indcs = np.nonzero(x0 > thrshld)[0]
    ez_pred[ez_indcs] = 1
    return ez_pred


def find_ez_x_fp(x, lb, ub):
    ez_pred = np.zeros_like(x)
    ez_indcs = np.nonzero(np.logical_and(x > lb, x < ub))[0]
    ez_pred[ez_indcs] = 1
    return ez_pred


def precision_recall(ez_hyp, ez_pred):
    tp = fp = fn = 0
    for a, b in zip(ez_hyp, ez_pred):
        if (a == 1 and b == 1):
            tp += 1
        elif (a == 1 and b == 0):
            fn += 1
        elif (a == 0 and b == 1):
            fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def find_recall(ez_hyp, ez_pred):
    tp = fp = fn = 0
    for a, b in zip(ez_hyp, ez_pred):
        if (a == 1 and b == 1):
            tp += 1
        elif (a == 1 and b == 0):
            fn += 1
        elif (a == 0 and b == 1):
            fp += 1
    recall = tp / (tp + fn)
    return recall