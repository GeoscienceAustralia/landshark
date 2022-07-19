landshark-import --nworkers 0 --batch-mb 0.001 tifs \
    --name sirsam \
    --ignore-crs  \
    --continuous ../integration/data/continuous

landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name Na_ppm_i_1 \
  --record Na_ppm_i_1 \
  --dtype continuous

landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
  --features features_sirsam.hdf5 \
  --split 1 10 \
  --targets targets_Na_ppm_i_1.hdf5 \
  --name sirsam \
  --halfwidth 1

landshark --keras-model train \
  --data traintest_sirsam_fold1of10 \
  --config nn_regression_keras.py \
  --epochs 20 \
  --iterations 50

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

parallel query_predict ::: {1..5} ::: 5
#1 5
#2 5
#3 5
#4 5
#5 5

# gdal_merge.py -o merged.tif nn_regression_keras_model_1of10/predictions_Na_ppm_i_1_*of4.tif

# merge statement
# echo gdal_merge.py -o predictions_log_cond_1.tif \\ > merge.txt
# for i in {1..384}; do echo nn_regression_keras_global_local_model_6of10/predictions_log_cond_1_${i}of384.tif >> merge.txt \\ ; done
# bash merge.txt

#for n in {1..N};
#  do echo $n;
#
#done
#
#landshark-extract --nworkers 0 --batch-mb 0.001 query \
#  --features features_sirsam.hdf5 \
#  --strip 5 10 --name sirsam \
#  --halfwidth 1
#
#landshark --keras-model --batch-mb 0.001 predict \
#  --config nn_regression_keras.py \
#  --checkpoint nn_regression_keras_model_1of10 \
#  --data query_sirsam_strip5of10
