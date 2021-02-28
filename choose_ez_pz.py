# %%
import numpy as np
import matplotlib.pyplot as plt
import lib.io.tvb
from tvb.simulator.lab import *
# %%
gain = np.loadtxt('datasets/syn_data/id001_bt/gain_inv-square.destrieux.txt')
cntrs,lbls = lib.io.tvb.read_roi_cntrs('datasets/syn_data/id001_bt/connectivity.destrieux.zip')
con = connectivity.Connectivity.from_file('/home/anirudh/Academia/projects/isp_benchmark/datasets/syn_data/id001_bt/connectivity.destrieux.zip')
con.weights[np.diag_indices(con.weights.shape[0])] = 0
con.weights = con.weights/np.max(con.weights)
# %%
gain_sum = gain.sum(axis=0)
gain_mean = gain.mean(axis=0)
gain_std = gain.std(axis=0)
# %%
plt.figure(figsize=(25,20))
plt.subplot(311)
plt.bar(range(gain.shape[1]), gain_sum);
plt.subplot(312)
plt.bar(range(gain.shape[1]), gain_mean);
plt.subplot(313)
plt.bar(range(gain.shape[1]), gain_std);

# %%
chsn_ez = np.argsort(gain.sum(axis=0))[-5:]
print(chsn_ez)
print([lbls[roi] for roi in chsn_ez])

# %%
ez_con = con.weights[chsn_ez]
# for roi in chsn_ez:
#     pz = con.weights[roi,:].argsort()[-3:]
#     print(f"ROI strongly connected to {roi}: {pz}")
chsn_pz = np.unique(ez_con.argsort(axis=1)[:, -2:].flatten())
chsn_pz = np.setdiff1d(chsn_pz, chsn_ez)
# chsn_pz = np.setdiff1d(chsn_pz, chsnez)
print(chsn_pz)
# print([lbls[roi] for roi in chsn_pz])