"""
I/O functions for working with CmdStan executables.

"""

import os
import subprocess
import numpy as np


def _rdump_array(key, val):
    c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
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


def rload(fname):
    """Load a dict of data from an R dump format file.
    """
    with open(fname, 'r') as fd:
        lines = fd.readlines()
    data = {}
    for line in lines:
        lhs, rhs = [_.strip() for _ in line.split('<-')]
        if rhs.startswith('structure'):
            *_, vals, dim = rhs.replace('(', ' ').replace(')', ' ').split('c')
            vals = [float(v) for v in vals.split(',')[:-1]]
            dim = [int(v) for v in dim.split(',')]
            val = np.array(vals).reshape(dim[::-1]).T
        elif rhs.startswith('c'):
            val = np.array([float(_) for _ in rhs[2:-1].split(',')])
        else:
            try:
                val = int(rhs)
            except:
                try:
                    val = float(rhs)
                except:
                    raise ValueError(rhs)
        data[lhs] = val
    return data


def merge_csv_data(*csvs, skip=0):
    data_ = {}
    for csv in csvs:
        for key, val in csv.items():
            val = val[skip:]
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
        csv = []
        for _ in fname:
            try:
                csv.append(parse_csv(_))
            except Exception as e:
                print('skipping ', fname, e) 
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
        new_shape = (-1,) + maxdims.get(name, ())
        data_[name] = data[:, idx].reshape(new_shape)

    return data_


def parse_summary_csv(fname):
    skeys = []
    svals = []
    with open(fname, 'r') as fd:
        scols = fd.readline().strip().split(',')
        for line in fd.readlines():
            if line.startswith('#'):
                break
            _, k, v = line.split('"')
            skeys.append(k)
            svals.append(np.array([float(_) for _ in v.split(',')[1:]]))
    svals = np.array(svals)

    sdat = {}
    sdims = {}
    for skey, sval in zip(skeys, svals):
        if '[' in skey:
            name, dim = skey.replace('[', ']').split(']')[:-1]
            dim = tuple(int(i) for i in dim.split(','))
            sdims[name] = dim
            if name not in sdat:
                sdat[name] = []
            sdat[name].append(sval)
        else:
            sdat[skey] = sval

    for key in [_ for _ in sdat.keys()]:
        if key in sdims:
            sdat[key] = np.array(sdat[key]).reshape(sdims[key] + (-1,))
    return scols, sdat 


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
    r_fname = r_fname or csv_fname + '.R'
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


def compile_model(stan_fname, cc='clang++'):
    path = os.path.abspath(os.path.dirname(stan_fname))
    name = stan_fname[:-5]
    target = os.path.join(path, name)
    proc = subprocess.Popen(
        ['make', target, f'CC={cc}'],
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


# TODO class to run modules instead of %%bash in ipynb
