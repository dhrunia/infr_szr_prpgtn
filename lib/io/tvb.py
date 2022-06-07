import zipfile
import numpy as np


def read_roi_cntrs(cntrs_zipfile):
    zf = zipfile.ZipFile(cntrs_zipfile)
    roi_cntrs = []
    roi_lbls = []
    with zf.open('centres.txt') as t:
        for line in t:
            roi_cntrs.append(line.decode('utf-8').strip().split(' ')[1:])
            roi_lbls.append(line.decode('utf-8').strip().split(' ')[0])
    roi_cntrs = np.array(roi_cntrs, dtype=float)
    return roi_cntrs, roi_lbls


def read_conn(conn_zip_path):
    zf = zipfile.ZipFile(conn_zip_path)

    with zf.open('weights.txt') as f:
        conn = np.loadtxt(f)
        conn = conn / conn.max()
        conn[np.diag_indices_from(conn)] = 0.0

    roi_cntrs = []
    roi_lbls = []
    with zf.open('centres.txt') as t:
        for line in t:
            roi_cntrs.append(line.decode('utf-8').strip().split(' ')[1:])
            roi_lbls.append(line.decode('utf-8').strip().split(' ')[0])
    roi_cntrs = np.array(roi_cntrs, dtype=float)

    return conn, roi_cntrs, roi_lbls
