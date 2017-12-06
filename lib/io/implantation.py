
"""
I/O routines related to the names & positions of electrodes implanted
in the head.

"""

import re


# TODO update for workflow.py

def contact_names_fd(fd):
    contacts = []
    for line in fd.readlines():
        line = line.decode('ascii')
        parts = [p.strip() for p in line.strip().split('=')]
        if len(parts)>1 and parts[1] == 'SEEG':
            name, idx = re.match("([A-Z]+[a-z]*[']*)([0-9]+)", parts[0]).groups()
            idx = int(idx)
            contacts.append((name, idx))
    return contacts


def contact_names(ades_fname):
    contacts = []
    with open(ades_fname, 'r') as fd:
        for line in fd.readlines():
            parts = [p.strip() for p in line.strip().split('=')]
            if len(parts)>1 and parts[1] == 'SEEG':
                name, idx = re.match("([A-Z]+[a-z]*[']*)([0-9]+)", parts[0]).groups()
                idx = int(idx)
                contacts.append((name, idx))
    return contacts


def contacts2bipolar(contacts):
    bipnames = []
    bipidx = []
    for i in range(len(contacts)-1):
        (ln, li), (rn, ri) = contacts[i:i+2]
        if ln != rn:
            continue
        bipnames.append('%s%d-%d' % (ln, li, ri))
        bipidx.append((i, i+1))
    return bipnames, bipidx



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
