import numpy as np
from pyts.metrics import dtw


def dtw_variation_delta(a, b, method="classic", **kwargs):
    distance, paths = dtw(a, b, return_path=True, dist="absolute", method=method)
    paths = paths.T
    # for each point in a find all aligned idxs in b (could be more than 1)
    alignments = np.split(paths[:, 1], np.unique(paths[:, 0], return_index=True)[1][1:])
    variation_delta = list()
    for a_idx, b_idxs in enumerate(alignments):
        variation_delta.append(np.abs(a[a_idx] - b[b_idxs].mean()))
    return np.array(variation_delta)
