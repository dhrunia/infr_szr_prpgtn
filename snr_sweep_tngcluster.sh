ROOT_DIR="${PWD}/..";

for snr in $(seq 0.1 0.1 2.5);
do
    for i in $(seq 1 1 10);
    do
        sbatch run_optim.sh \
        "${ROOT_DIR}/Rfiles/fit_data_snsrfit_ode_snr${snr}_sample${i}.R" \
        "${ROOT_DIR}/Rfiles/param_init.R" \
        "${ROOT_DIR}/samples/samples_snr${snr}_sample${i}.csv" \
        "${ROOT_DIR}/logs/snsrfit_ode_snr${snr}_sample${i}.log";
    done
done
