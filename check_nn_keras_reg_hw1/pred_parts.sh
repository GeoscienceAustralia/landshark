#!/bin/bash
pid=$(grep ^Pid /proc/self/status)
corelist=$(grep Cpus_allowed_list: /proc/self/status | awk '{print $2}')
host=$(hostname | sed 's/.gadi.nci.org.au//g')
echo subtask $1 running in $pid using cores $corelist on compute node $host


module load python3/3.9.2 gdal/3.5.0
source /g/data/ge3/sudipta/venvs/nompiland/bin/activate
export PYTHONPATH=/apps/gdal/3.5.0/lib/python3.9/site-packages:/g/data/ge3/sudipta/venvs/nompiland/lib/python3.9/site-packages/:/apps/python3/3.9.2/lib/python3.9/site-packages/


function query_predict {
    echo starting query and preict $1 of $2
    n=$1
    N=$2
    landshark-extract --nworkers 0 --batch-mb 0.1 query \
        --features ./features_wa_cond.hdf5 \
        --strip ${n} ${N} --name wa_cond \
        --halfwidth 1
    landshark --keras-model --batch-mb 0.1 predict \
        --config ./nn_regression_keras_global_local.py \
        --checkpoint ./nn_regression_keras_global_local_model_6of10 \
        --data query_wa_cond_strip${n}of${N}
    echo done query and preict $1 of $2
}
export -f query_predict

# create the target directory
mkdir query_wa_cond_strip${1}of${2}

# run the query and prediction function, but redirect logs to it's own directory
query_predict $1 $2 > query_wa_cond_strip${1}of${2}/$PBS_JOBNAME__$PBS_JOBID_$1of$2.log 2>&1
