ROOT_DIR="$(echo "$(cd ../ && pwd)")"

for sigma_prior in $(seq 0.1 0.1 1.0);
do
    for i in $(seq 1 1 10);
    do
    sbatch run_optim.sh \
        "${ROOT_DIR}/Rfiles/fit_data_snsrfit_ode.R" \
        "${ROOT_DIR}/Rfiles/param_init_sigmaprior${sigma_prior}_sample${i}.R" \
        "${ROOT_DIR}/samples/samples_sigmaprior${sigma_prior}_sample${i}.csv" \
        "${ROOT_DIR}/logs/snsrfit_ode_sigmaprior${sigma_prior}_sample${i}.log";    
    done
done
