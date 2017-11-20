"""
Library of useful routines for virtual epileptic patient workflows. 

"""

import os
import re
import sys
import time
import random
import tarfile
import subprocess
import numpy as np
import pylab as pl

try:
    import nibabel
except ImportError:
    import pip; pip.main('install nibabel'.split())
    import nibabel


def log_power_change(fs, seeg, flo=10.0, nfft=1024, tb=10.0):
    pl.ioff()
    ps = []
    for chan in seeg:
        P, F, T, _ = pl.specgram(chan, NFFT=nfft, Fs=fs)
        p = P[F > flo].sum(axis=0)
        p /= p[T < tb].mean()
        ps.append(pl.detrend_linear(np.log(p)))
    return T, np.array(ps)


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


def load_centers(centers_fname):
    reg_xyz = loadtxt('data/centers.txt', usecols=(1,2,3))
    reg_names = []
    with open('data/centers.txt', 'r') as fd:
        for i, line in enumerate(fd.readlines()):
            reg_names.append('r%03d-%s' % (i, line.strip().split()[0]))
    seg = nibabel.load('data/aparcaseg_2_diff_2.nii.gz')
    reg_xyz = seg.affine.dot(c_[reg_xyz, ones((reg_xyz.shape[0], ))].T)[:3].T
    return reg_xyz


def load_seeg_labels(elecs_fname):
    seeg_labels = []
    seeg_xyz = loadtxt('data/elecs_name.txt', usecols=(1,2,3))
    with open('data/elecs_name.txt', 'r') as fd:
        for line in fd.readlines():
            parts = line.strip().split()
            seeg_labels.append(parts[0].upper())
    #print sorted(seeg_labels+[e+str(i) for e, i in contacts])
    return seeg_labels


def seeg_xyz():
    monopolar_chan_to_pos = []
    for name, idx in contacts:
        monopolar_chan_to_pos.append(seeg_labels.index(name+str(idx)))
    seeg_xyz = array([seeg_xyz[i] for i in monopolar_chan_to_pos])
    return seeg_xyz


def prep_stan_data():
    # load sEEG, pick channels, analyze
    seeg = load_complex()
    fn = 'seedpsd.npz'
    if not os.path.exists(fn):
        print('recomputing log power change')
        T, ps = log_power_change(512.0, seeg, nfft=512)
        np.savez(fn, T=T, ps=ps)
    npz = np.load(fn)
    T, ps = npz['T'], npz['ps']
    picks = r_[78, 14] # r_[78, 23, 54, 14, 35, 100]
    T -= T[0]
    dt = T[1] - T[0]
    print('dt based on psd is ', dt)
    def plot_ps():
        figure()
        plot(T, ps[picks].T + r_[:len(picks)]*10, 'k-x', linewidth=0.5)
        yticks(r_[:len(picks)]*10, [str(_) for _ in picks])
        savefig('prepdata-ps.png')
    plot_ps()
    # lead field, select nodes
    Vr = np.load('data/Vr.npy')
    nodes = []#158, 157, 147]
    for i in picks:
        pick_nodes = argsort(Vr[i])[-3:].tolist()
        nodes.extend(pick_nodes)
        print(f"{i} {contacts[i]} picks")
        for pick_node in pick_nodes:
            print(f'\t{reg_names[pick_node]}')
    nodes = unique(nodes)
    Vr_ = Vr[picks][:, nodes]
    Vr_ /= Vr_.sum(axis=1)[:, newaxis]
    print('reduced gain matrix shape is ', Vr_.shape)
    # reduce connectivity & compute appropriate constant
    W = loadtxt('data/weights.txt')
    iconst = setxor1d(r_[:W.shape[0]], nodes)
    for i in range(W.shape[0]):
        W[i, i] = 0.0
    W /= W.max()
    W_ = W[nodes][:, nodes]
    Ic = W[nodes][:, iconst].sum(axis=1)
    # build dataset
    tm = c_[T>100.0, T<450.0].all(axis=1)
    T = T[tm]
    print (ps.shape)
    ps = ps[picks][:, tm]
    print (ps.shape)
    data_simple = {
        'nn': W_.shape[0], 'ns': Vr_.shape[0], 'nt': len(T),
        'I1': 3.1, 'tau0': 3.0, 'dt': dt,
        'SC': W_, 'SC_var': 5.0, 'gain': Vr_, 'seeg_log_power': ps.T, 'Ic': Ic,
        'K_lo': 1.0, 'K_u': 5.0, 'K_v': 10.0,
        'x0_lo': -15.0, 'x0_hi': 5.0, 'eps_hi': 0.2, 'sig_hi': 0.025,
        'zlim': r_[0.0, 10.0],
        'siguv': r_[-1.0, 0.5],
        'epsuv': r_[-1.0, 0.5],
        'use_data': 1,
        'tt': 0.08
    }
    npz = {'nodes': nodes}
    npz.update(data_simple)
    np.savez('data.R.npz', **npz)
    rdump('data.R', data_simple, )


def _rdump_array(key, val):
    c = 'c(' + ', '.join(map(str, val.flat)) + ')'
    if (val.size,) == val.shape:
        return '{key} <- {c}'.format(key=key, c=c)
    else:
        dim = '.Dim = c{0}'.format(val.shape)
        struct = '{key} <- structure({c}, {dim})'.format(
            key=key, c=c, dim=dim)
        return struct


def rdump(fname, data):
    """Dump a dict of data to a R dump format file.
    """
    with open(fname, 'w') as fd:
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.size > 1:
                line = _rdump_array(key, val)
            else:
                try:
                    val = val.flat[0]
                except:
                    pass
                line = '%s <- %s' % (key, val)
            fd.write(line)
            fd.write('\n')


def merge_csv_data(*csvs):
    data_ = {}
    for csv in csvs:
        for key, val in csv.items():
            if key in data_:
                data_[key] = np.concatenate(
                    (data_[key], val),
                    axis=0
                )
            else:
                data_[key] = val
    return data_


def parse_csv(fname, merge=True):
    if '*' in fname:
        import glob
        return parse_csv(glob.glob(fname), merge=merge)
    if isinstance(fname, (list, tuple)):
        csv = [parse_csv(_) for _ in fname]
        if merge:
            csv = merge_csv_data(*csv)
        return csv
    
    lines = []
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            if not line.startswith('#'):
                lines.append(line.strip().split(','))
    names = [field.split('.') for field in lines[0]]
    data = np.array([[float(f) for f in line] for line in lines[1:]])

    namemap = {}
    maxdims = {}
    for i, name in enumerate(names):
        if name[0] not in namemap:
            namemap[name[0]] = []
        namemap[name[0]].append(i)
        if len(name) > 1:
            maxdims[name[0]] = name[1:]

    for name in maxdims.keys():
        dims = []
        for dim in maxdims[name]:
            dims.append(int(dim))
        maxdims[name] = tuple(reversed(dims))

    # data in linear order per Stan, e.g. mat is col maj
    # TODO array is row maj, how to distinguish matrix v array[,]?
    data_ = {}
    for name, idx in namemap.items():
        new_shape = (-1, ) + maxdims.get(name, ())
        data_[name] = data[:, idx].reshape(new_shape)

    return data_


def csv2mode(csv_fname, mode=None):
    csv = parse_csv(csv_fname)
    data = {}
    for key, val in csv.items():
        if key.endswith('__'):
            continue
        if mode is None:
            val_ = val[0]
        elif mode == 'mean':
            val_ = val.mean(axis=0)
        elif mode[0] == 'p':
            val_ = np.percentile(val, int(mode[1:]), axis=0)
        data[key] = val_
    return data


def csv2r(csv_fname, r_fname=None, mode=None):
    data = csv2mode(csv_fname, mode=mode)
    r_fname = r_fname or csv_fname.replace('.csv', '.R')
    rdump(r_fname, data)
    
    
class CmdStanNotFound(Exception): pass


def cmdstan_path(path=''):
    if path:
        path = os.path.expanduser(os.path.expandvars(path))
        os.environ['CMDSTAN'] = path
    path = os.environ.get('CMDSTAN', 'cmdstan')
    if not os.path.exists(os.path.join(path, 'runCmdStanTests.py')):
        raise CmdStanNotFound(
            'please provide CmdStan path, e.g. lib.cmdstan_path("/path/to/")')
    return path

def compile_model(stan_fname):
    path = os.path.abspath(os.path.dirname(stan_fname))
    name = stan_fname[:-5]
    target = os.path.join(path, name)
    proc = subprocess.Popen(
        ['make', target],
        cwd=cmdstan_path(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = proc.stdout.read().decode('ascii').strip()
    if stdout:
        print(stdout)
    stderr = proc.stderr.read().decode('ascii').strip()
    if stderr:
        print(stderr)
    
    
def viz_phase_space(data):
    opt = len(data['x']) == 1
    npz = np.load('data.R.npz')
    tr = lambda A: np.transpose(A, (0, 2, 1))
    x, z = tr(data['x']), tr(data['z'])
    tau0 = npz['tau0']
    X, Z = np.mgrid[-5.0:5.0:50j, -5.0:5.0:50j]
    dX = (npz['I1'] + 1.0) - X**3.0 - 2.0*X**2.0 - Z
    x0mean = data['x0'].mean(axis=0)
    Kmean = data['K'].mean(axis=0)
    def nullclines(i):
        pl.contour(X, Z, dX, 0, colors='r')
        dZ = (1.0/tau0) * (4.0 * (X - x0mean[i])) - Z - Kmean*(-npz['Ic'][i]*(1.8 + X))
        pl.contour(X, Z, dZ, 0, colors='b')
    for i in range(x.shape[-1]):
        pl.subplot(2, 3, i + 1)
        if opt:
            pl.plot(x[0, :, i], z[0, :, i], 'k', alpha=0.5)
        else:
            for j in range(1 if opt else 10):
                pl.plot(x[-j, :, i], z[-j, :, i], 'k', alpha=0.2, linewidth=0.5)
        nullclines(i)
        pl.grid(True)
        pl.xlabel('x(t)')
        pl.ylabel('z(t)')
        pl.title(f'node {i}')
    pl.tight_layout()


def viz_pair_plots(csv, keys, skip=0):
    n = len(keys)
    if isinstance(csv, dict):
        csv = [csv]  # following assumes list of chains' results
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            pl.subplot(n, n, i*n+j+1)
            for csvi in csv:
                if i==j:
                    pl.hist(csvi[key_i][skip:], 20, log=True)
                else:
                    pl.plot(csvi[key_j][skip:], csvi[key_i][skip:], '.')
            if i==0:
                pl.title(key_j)
            if j==0:
                pl.ylabel(key_i)
    pl.tight_layout()

# TODO run w/ optimize before


def reload():
    import importlib, lib
    eval('importlib.reload(lib)')
