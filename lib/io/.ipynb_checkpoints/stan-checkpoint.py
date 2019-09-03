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
            if '"' not in line:
                continue
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



def rem_warmup_samples(src_fname,trgt_fname,num_warmup_samples):
    with open(src_fname,'r') as fd1:
        with open(trgt_fname,'w') as fd2:
            while(True):
                t = fd1.readline()
                if(t[0] == '#'):
                    fd2.write(t)
                else:
                    fd2.write(t)
                    for i in range(num_warmup_samples):
                        fd1.readline()
                    break
            t = fd1.readline()
            while(t):
                fd2.write(t)
                t = fd1.readline()


# def merge_csv(csvs, out_csv):
#     with open(out_csv, 'w') as fd_out_csv:
#         for i in len(csvs):
#             if(i == 1):
#                 with open(csvs[i], 'r') as fd:
#                     for line in fd:
#                         fd_out_csv.write(line)
#             else:
#                 with open(csvs[i], 'r') as fd:
#                     for line in fd:
#                         if(line[0] != '#'):
#                             break
#                     for line in fd:
#                         if(line[0] != '#'):
#                             fd_out_csv.write(line)


def read_one_sample(line, data, sample_idx, var_names, var_dims,
                    var_start_idx):
    var_vals = [float(el.strip()) for el in line.split(',')]
    for var_name in var_names:
        start_idx = var_start_idx[var_name]
        end_idx = start_idx + (np.product(var_dims[var_name])
                               if (var_dims[var_name]) else 1)
        data[var_name][sample_idx] = np.array(
            var_vals[start_idx:end_idx]).reshape(
                var_dims[var_name], order='F')


def read_samples(csvs, nwarmup=0, nsampling=0, variables_of_interest=[]):
    '''
    Reads specified variables from cmdstan output csv file
    csv_fname : cmdstan's output csv file name
    nwarmup   : Number of warmup samples to read
    nsampling : Number of samples after warmup
    variables_of_interest : list of variable names to read
    '''
    samples_chain = []
    for csv_fname in csvs:
        ignore_warmup = False if (nwarmup) else True
        nsamples = nwarmup + nsampling
        with open(csv_fname, 'r') as fd:
            t = fd.readline().strip()
            read_head = False
            sample_idx = 0
            sample_count = 0
            while (t):
                if (t[0] == '#'):
                    if ('method' in t):
                        algorithm = [
                            el.strip() for el in t[1:].strip().split("=")
                        ][-1]
                        if('optimize' in algorithm):
                            nsamples = 1
                    elif ('num_samples' in t):
                        sampling_iters = int(
                            t[1:].strip().split(' ')[2].strip())
                        if (nsampling > sampling_iters):
                            raise (Exception(
                                'nsampling cannot be greater than the number of sampling \
    iterations'))
                    elif ('num_warmup' in t):
                        warmup_iters = int(t[1:].strip().split(' ')[2].strip())
                        if (nwarmup > warmup_iters):
                            raise (Exception(
                                'nwarmup cannot be greater than the number of warmup \
    iterations'))
                        if (nsamples == 0):
                            nsamples = sampling_iters + 0 if (
                                ignore_warmup) else warmup_iters
                    elif ('save_warmup' in t):
                        save_warmup = int(t[1:].strip().split(' ')[2].strip())
                        if (not ignore_warmup and not save_warmup):
                            raise (Exception(
                                'csv file does not contain warmup samples, nwarmup must be \
    zero'))
                    elif ('output_samples' in t):
                        nsamples = int([
                            el.strip() for el in t[1:].strip().split("=")
                        ][-1].split()[0])
                elif (
                        not read_head
                ):  # Extract variable names and their dimensions from the heading of the csv
                    var_names = []
                    var_dims = {}
                    var_start_idx = {}
                    col_names = t.split(',')
                    for i, name in enumerate(col_names):
                        var_name = name.strip().split('.')[0]
                        var_dim = [
                            int(dim) for dim in name.strip().split('.')[1:]
                        ]
                        if (var_name not in var_names):
                            var_names.append(var_name)
                            var_start_idx[var_name] = i
                        var_dims[var_name] = var_dim
                    read_head = True
                    data = {}
                    var_names = variables_of_interest if (
                        variables_of_interest) else var_names
                    # Create a dictionary (variable name -> numpy.ndarray) for storing data
                    for var_name in var_names:
                        data[var_name] = np.ndarray(
                            shape=[nsamples] + var_dims[var_name], dtype=float)
                else:
                    if ('sample' in algorithm):
                        if (ignore_warmup and save_warmup
                                and sample_count < warmup_iters):
                            pass
                        elif (not ignore_warmup and sample_count < nwarmup):
                            read_one_sample(t, data, sample_idx, var_names,
                                            var_dims, var_start_idx)
                            sample_idx += 1
                        elif (not ignore_warmup
                              and sample_count < warmup_iters):
                            pass
                        else:
                            read_one_sample(t, data, sample_idx, var_names,
                                            var_dims, var_start_idx)
                            sample_idx += 1
                    elif ('variational' in algorithm):
                        read_one_sample(t, data, sample_idx, var_names,
                                        var_dims, var_start_idx)
                        sample_idx += 1
                    elif ('optimize' in algorithm):
                        read_one_sample(t, data, sample_idx, var_names,
                                        var_dims, var_start_idx)
                        sample_idx += 1
                    if (sample_idx == nsamples):
                        break
                    sample_count += 1
                t = fd.readline()
        samples_chain.append(data)
    samples = {}
    for key in samples_chain[0].keys():
        samples[key] = np.concatenate(
            [samples_chain[i][key] for i in range(len(samples_chain))])
    return samples


def read_last_sample(csv_fname, variables_of_interest=[]):
    with open(csv_fname, 'r') as fd:
        read_head = False
        t = fd.readline()
        while(t):
            # print(t[0:10])
            if(t[0] == '#'):
                pass
            elif(not read_head):
                var_names = []
                var_dims = {}
                var_start_idx = {}
                col_names = t.split(',')
                for i, name in enumerate(col_names):
                    var_name = name.strip().split('.')[0]
                    var_dim = [
                        int(dim) for dim in name.strip().split('.')[1:]
                    ]
                    if (var_name not in var_names):
                        var_names.append(var_name)
                        var_start_idx[var_name] = i
                    var_dims[var_name] = var_dim
                read_head = True
                data = {}
                var_names = variables_of_interest if (
                    variables_of_interest) else var_names
                # Create a dictionary (variable name -> numpy.ndarray) for storing data
                for var_name in var_names:
                    data[var_name] = np.ndarray(
                        shape=[1] + var_dims[var_name], dtype=float)
            else:
                read_one_sample(t, data, 0, var_names, var_dims, var_start_idx)
            t = fd.readline()
    # Remove the indexing for samples as it has only one sample
    for key in data.keys():
        data[key] = data[key][0]
    return data


def read_metric(csv_fname):
    adapt_completed = False
    try:
        with open(csv_fname) as fd:
            line = fd.readline()
            while (line):
                if (line[0] == '#' and 'adaptation terminated' in line.lower()):
                    adapt_completed = True
                    break
                line = fd.readline()
            if(adapt_completed):
                line = fd.readline()
                step_size = float(line.strip().split('=')[-1].strip())
                line = fd.readline()
                line = fd.readline()
                inv_mass_mat = np.array(
                    [float(el.strip()) for el in line[1:].strip().split(',')])
                return (step_size, inv_mass_mat)
            else:
                return (None, None)
    except IOError as err:
        print(err)
        return (None, None)
