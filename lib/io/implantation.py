
"""
I/O routines related to the names & positions of electrodes implanted
in the head.

"""


import os
import re
from typing import Dict
import numpy as np


def load_contact_positions(fname: str) -> Dict[str, np.ndarray]:
    """
    Read contact names as identified from an anatomical scan containing the
    contacts.

    :param fname: file w/ lines like "GPH'1 3.04 45.64 -32.0" in FS T1 coords
    :return: map of name to xyz position as array
    """
    contact_positions = {}
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            name_idx, x, y, z = line.strip().split()
            xyz = np.r_[float(x), float(y), float(z)]
            contact_positions[name_idx] = xyz
    return contact_positions


def load_ades_merge_positions(fname: str,
                              contact_positions: dict
                              ) -> (dict, np.ndarray):
    """
    Reads contact names as found in recorded time series in ADES format,
    and merge with contact positions information.

    :param fname: ADES header file name
    :param contact_positions: dict returned by load_contact_positions.
    :return: merged dictionary and (n, 3) array of contact positions
    """
    ades_contact_re = r"([A-Z]+[a-z]*[']*)([0-9]+)"
    contacts = []
    samp_rate = None
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            if line.strip().startswith('#'):
                continue
            key, val = [p.strip() for p in line.strip().split('=')]
            if key == 'samplingRate':
                samp_rate = float(val)
            if val == 'SEEG':
                name, idx = re.match(ades_contact_re, key).groups()
                idx = int(idx)
                contacts.append((name, idx, contact_positions[key.lower()]))
    cxyz = np.array([xyz for _, _, xyz in contacts])


# def contact_names_fd(fd):
#     contacts = []
#     for line in fd.readlines():
#         line = line.decode('ascii')
#         parts = [p.strip() for p in line.strip().split('=')]
#         if len(parts)>1 and parts[1] == 'SEEG':
#             name, idx = re.match("([A-Z]+[a-z]*[']*)([0-9]+)", parts[0]).groups()
#             idx = int(idx)
#             contacts.append((name, idx))
#     return contacts
#
#
# def contact_names(ades_fname):
#     contacts = []
#     with open(ades_fname, 'r') as fd:
#         for line in fd.readlines():
#             parts = [p.strip() for p in line.strip().split('=')]
#             if len(parts)>1 and parts[1] == 'SEEG':
#                 name, idx = re.match("([A-Z]+[a-z]*[']*)([0-9]+)", parts[0]).groups()
#                 idx = int(idx)
#                 contacts.append((name, idx))
#     return contacts
#
#
# def contacts2bipolar(contacts):
#     bipnames = []
#     bipidx = []
#     for i in range(len(contacts)-1):
#         (ln, li), (rn, ri) = contacts[i:i+2]
#         if ln != rn:
#             continue
#         bipnames.append('%s%d-%d' % (ln, li, ri))
#         bipidx.append((i, i+1))
#     return bipnames, bipidx



# def load_seeg_labels(elecs_fname):
#     seeg_labels = []
#     seeg_xyz = np.loadtxt('data/elecs_name.txt', usecols=(1,2,3))
#     with open('data/elecs_name.txt', 'r') as fd:
#         for line in fd.readlines():
#             parts = line.strip().split()
#             seeg_labels.append(parts[0].upper())
#     #print sorted(seeg_labels+[e+str(i) for e, i in contacts])
#     return seeg_labels


# def seeg_xyz():
#     monopolar_chan_to_pos = []
#     for name, idx in contacts:
#         monopolar_chan_to_pos.append(seeg_labels.index(name+str(idx)))
#     seeg_xyz = array([seeg_xyz[i] for i in monopolar_chan_to_pos])
#     return seeg_xyz
