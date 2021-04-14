ROOT_DIR="$(echo "$(cd ../ && pwd)")"

for nez in $(seq 3 1 5);
do
    for snr in $(seq 0.1 0.1 2.5);
    do
        for i in $(seq 1 1 10);
        do
        sbatch run_optim.sh \
            "${ROOT_DIR}/Rfiles/fit_data_snsrfit_ode_${nez}ez_snr${snr}_sample${i}.R" \
            "${ROOT_DIR}/Rfiles/param_init_${nez}ez.R" \
            "${ROOT_DIR}/samples/samples_${nez}ez_snr${snr}_sample${i}.csv" \
            "${ROOT_DIR}/logs/snsrfit_ode_${nez}ez_snr${snr}_sample${i}.log";
        done
    done
done
