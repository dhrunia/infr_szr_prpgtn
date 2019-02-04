import subprocess

# compile the stan code
try:
    proc = subprocess.run('/home/anirudhnihalani/scripts/stancompile.sh vep-snsrfit-ode',
                          check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=True)
    print(proc.stdout.decode('UTF-8'))
    proc = subprocess.run('/home/anirudhnihalani/scripts/stancompile.sh vep-snsrfit-ode-norm-prior',
                          check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=True)
    print(proc.stdout.decode('UTF-8'))
except subprocess.CalledProcessError as error:
    print(error.stderr.decode('UTF-8'))

# submit jobs
for hyp_type in ['ez', 'pz', 'hz']:
    cmd1 = f'sbatch run.sh vep-snsrfit-ode 2000 1000 results/exp12 4 15 0.95 fit_data_{hyp_type}.R {hyp_type} lognormpriors'
    cmd2 = f'sbatch run.sh vep-snsrfit-ode-norm-prior 2000 1000 results/exp12 4 15 0.95 fit_data_{hyp_type}.R {hyp_type} normpriors'
    subprocess.Popen(cmd1.split())
    subprocess.Popen(cmd2.split())
