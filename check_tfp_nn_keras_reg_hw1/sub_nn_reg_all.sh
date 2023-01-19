set -e
landshark-import --nworkers 0 --batch-mb 0.001 tifs \
    --name sirsam \
    --ignore-crs  \
    --continuous ../integration/data/continuous

landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name sirsam \
  --record Na_ppm_i_1 \
  --dtype continuous

landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
  --features features_sirsam.hdf5 \
  --split 1 10 \
  --targets targets_sirsam.hdf5 \
  --name sirsam \
  --halfwidth 1


## proba = True
landshark --keras-model train \
  --trainvalidation false \
  --data traintest_sirsam_fold1of10/ \
  --config nn_regression_keras_global_local.py  \
  --epochs 10 --batchsize 153 --iterations 10
#
landshark-extract --batch-mb 0.01 query \
    --features features_sirsam.hdf5 \
    --strip 5 200 \
    --name sirsam \
    --halfwidth 1

landshark -v DEBUG --keras-model --batch-mb 0.001 predict \
    --proba true \
    --config nn_regression_keras_global_local.py \
    --checkpoint nn_regression_keras_global_local_model_1of10 \
    --data query_sirsam_strip5of200 \
    --pred_ensemble_size 12

# non-proba
#landshark --keras-model train \
#  --trainvalidation false \
#  --data traintest_sirsam_fold1of10/ \
#  --config nn_regression_keras.py  \
#  --epochs 10 \
#  --iterations 53 --batchsize 100
#
#landshark-extract --batch-mb 0.01 query \
#    --features features_sirsam.hdf5 \
#    --strip 5 200 \
#    --name sirsam \
#    --halfwidth 0
#
#landshark --keras-model --batch-mb 0.01 predict \
#    --proba false \
#    --config nn_regression_keras.py \
#    --checkpoint nn_regression_keras_model_1of10 \
#    --data query_sirsam_strip5of200

# oos validation:
landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name sirsam_oos \
  --record Na_ppm_i_1 \
  --dtype continuous


# TODO: can avoid saving one of train.xxx.tfrecord and test.xxx.tfrecord
landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
  --features features_sirsam.hdf5 \
  --split 1 1 \
  --targets targets_sirsam_oos.hdf5 \
  --name sirsam_oos \
  --halfwidth 1

landshark -v DEBUG --keras-model --batch-mb 0.001 predict_oos \
    --proba true \
    --config nn_regression_keras_global_local.py \
    --checkpoint nn_regression_keras_global_local_model_1of10 \
    --data traintest_sirsam_oos_fold1of1 \
    --pred_ensemble_size 12