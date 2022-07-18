#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=10GB
#PBS -l jobfs=10GB
#PBS -q normal
#PBS -P ge3
#PBS -l walltime=20:00:00
#PBS -l storage=gdata/dg9+gdata/dz56+gdata/ge3
#PBS -l wd
#PBS -j oe

module load tensorflow/2.6.0  python3/3.9.2 openmpi/4.1.1 gdal/3.0.2 parallel
source /g/data/ge3/sudipta/venvs/land3p9n/bin/activate
export PYTHONPATH=/app/gdal/3.0.2/lib64:/apps/tensorflow/2.6.0/lib/python3.9/site-packages:/g/data/ge3/sudipta/venvs/land3p9n/lib/python3.9/site-packages/:/apps/python3/3.9.2/lib/python3.9/site-packages/

function query_predict {
    echo starting query and preict $1 of $2
    n=$1
    N=$2
    landshark-extract --nworkers 0 --batch-mb 0.001 query \
        --features features_sirsam.hdf5 \
        --strip ${n} ${N} --name sirsam \
        --halfwidth 1
    landshark --keras-model --batch-mb 0.001 predict \
        --config nn_regression_keras.py \
        --checkpoint nn_regression_keras_model_1of10 \
        --data query_sirsam_strip${n}of${N}
    echo done query and preict $1 of $2
}
export -f query_predict
query_predict $1 $2
