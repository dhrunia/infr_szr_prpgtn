import zipfile
import numpy as np

def read_roi_cntrs(cntrs_zipfile):
    '''
    Read ROI centers, labels from the TVB connecitivty zip file

    Parameters:
    ----------
    cntrs_zipfile : zipfile
        TVB connectivity zip file

    Returns:
    -------
    Centers : list
        Talairach ROI centers
    
    Labels : list
        ROI names/labels
    '''
    zf = zipfile.ZipFile(cntrs_zipfile)
    roi_cntrs = []
    roi_lbls = []
    with zf.open('centres.txt') as t:
        for line in t:
            roi_cntrs.append(line.decode('utf-8').strip().split(' ')[1:])
            roi_lbls.append(line.decode('utf-8').strip().split(' ')[0])
    roi_cntrs = np.array(roi_cntrs, dtype=float)
    return roi_cntrs, roi_lbls
