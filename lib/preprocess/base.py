"""
Common preprocessing elements.

"""

import numpy as np


def bipolar_info(contacts: dict):
    """
    Constructs bipolar montage information from contact information.

    :param contacts: dict output of lib.io.implantation.load_*_merge_positions
    :return: tuple of bipolar contacts, bipolar projection matrix & positions.
    """
    contacts_bip = []
    cxyz_bip = []
    for i in range(len(contacts) - 1):
        (ln, li, lp), (rn, ri, rp) = contacts[i:i + 2]
        if ln != rn:
            continue
        xyzi = (lp + rp) / 2
        contacts_bip.append(('%s%d-%d' % (ln, li, ri), (i, i + 1), xyzi))
        cxyz_bip.append(xyzi)
    proj_bip = np.zeros((len(contacts_bip), len(contacts)))
    for i, (_, idx, _) in enumerate(contacts_bip):
        proj_bip[i, idx] = 1, -1
    return contacts_bip, proj_bip, np.array(cxyz_bip)


def simple_gain(cxyz, rxyz, aparcaseg=None, proj_bip=None):
    if aparcaseg:
        # TODO if we have aa, we can compute gain over all
        # TODO voxels and average
        pass
    gain = 1 / np.sqrt(np.sum((cxyz[:, None] - rxyz) ** 2, axis=2))
    if proj_bip:
        gain_bip = proj_bip.dot(gain)
        return gain, gain_bip
    return gain


def exclude_idx(contacts, names):
    if isinstance(names, str):
        names = names.split()
    idx = np.array([i for i, (name, *_) in enumerate(contacts)
                    if name in names])
    return idx


class BasePreproc:
    """
    Preprocessing takes a directory of patient data and produces a
    dataset suitable for Stan models under consideration.

    """

    # TODO coordinate data variable names across models
