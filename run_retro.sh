ROOT_DATA_DIR=$PWD/..
ROOT_RES_DIR=$PWD/..
STAN_FNAME=vep-snsrfit-ode-rescaled-nointerp;
SAMPLING_ITERS=200;
WARMUP_ITERS=200;
NCHAINS=12;
DELTA=0.95;
MAX_DEPTH=15;
JITTER=0;



for PATIENT_ID in $(ls ${ROOT_DATA_DIR} | grep -i id*);
do
    RES_DIR=${ROOT_RES_DIR}/${PATIENT_ID}/results;
    LOG_DIR=${ROOT_RES_DIR}/${PATIENT_ID}/logs;
    # mkdir -p ${RES_DIR};
    INIT_PATH=${ROOT_DATA_DIR}/${PATIENT_ID}/Rfiles/params_init.R
    for SZR_FNAME in $(ls ${ROOT_DATA_DIR}/${PATIENT_ID}/Rfiles/ | grep -i obs_data.*\.R);
    do
	DATA_PATH=${ROOT_DATA_DIR}/${PATIENT_ID}/Rfiles/${SZR_FNAME}
	SZR_NAME=$(echo $SZR_FNAME | cut -d '.' -f 1 | cut -d '_' -f 3-);
	sbatch run_hmc.sh ${DATA_PATH} ${INIT_PATH} ${RES_DIR} ${STAN_FNAME} ${SAMPLING_ITERS} ${WARMUP_ITERS} ${NCHAINS} ${DELTA} ${MAX_DEPTH} ${JITTER} ${SZR_NAME} ${LOG_DIR};
    done
done
