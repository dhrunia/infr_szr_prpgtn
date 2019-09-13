"""
Various utility functions for working with CmdStan
"""

import subprocess
import sys


def create_process(cmd,
                   block=None,
                   stdout=sys.stdout,
                   stderr=sys.stderr,
                   shell=False):
    if (block):
        subProc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        while subProc.poll() is None:
            stdout.write(subProc.stdout.read(1))
            stdout.flush()
        stdout.write(subProc.stdout.read())
        stdout.flush()
        err = subProc.stderr.read()
        if (err):
            stderr.write(err)
            stderr.flush()
            t = input('Continue[y/n]:')
            if (t == 'y'):
                return -1
            else:
                raise Exception("Error executing " + ' '.join(cmd))
        return 0
    else:
        subProc = subprocess.Popen(cmd, shell=shell)
        return subProc


def is_completed(log_fname):
    """
    Checks if a HMC run has finished warmup and sampling iterations
    Input:
    log_fname: File name containing output of the sampler
    Output:
    Returns True if sampling is complete, False otherwise
    """
    try:
        with open(log_fname) as fd:
            tail_log = fd.readlines()[-10:]
        if (any(['elapsed time' in line.lower() for line in tail_log])):
            return True
        else:
            return False
    except IOError as err:
        print(err)


def compile(stan_dir, stan_fname):
    """
    Wrapper function to compile stan files to executables
    Input:
    stan_dir -> STAN source directory
    stan_fname -> name of the file to compile (without .stan extension)
    """
    cmd = f'curr_dir=$(pwd); cd {stan_dir}; make $curr_dir/{stan_fname}'
    print(f'compiling {stan_fname}')
    compile_output = subprocess.check_output(
        cmd, shell=True, executable='/bin/bash', stderr=subprocess.STDOUT)
    print(compile_output.decode("utf-8"))
