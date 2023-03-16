set -e

# extract features
landshark-import --nworkers 0 --batch-mb 0.001 tifs \
    --name sirsam \
    --ignore-crs  \
    --continuous ../integration/data/continuous

# extract targets
landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name Na_ppm_i_1 \
  --record Na_ppm_i_1 \
  --dtype continuous

# create train/test data split
landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
  --features features_sirsam.hdf5 \
  --split 1 10 \
  --targets targets_Na_ppm_i_1.hdf5 \
  --name sirsam \
  --halfwidth 1

# extract validation data - note I am using the same targets as targets and validation here
# targets and validation hdf5 files could be different with a different validation shapefile/hdf5
landshark-extract --nworkers 0 --batch-mb 0.001 trainvalidate \
  --features features_sirsam.hdf5 \
  --targets targets_Na_ppm_i_1.hdf5 \
  --validation targets_Na_ppm_i_1.hdf5 \
  --name sirsam \
  --halfwidth 1


# train using train test split
landshark --keras-model train \
  --trainvalidation false \
  --data traintest_sirsam_fold1of10/ \
  --config nn_regression_keras.py  \
  --epochs 20 \
  --iterations 5

# train using validation data
landshark --keras-model train \
  --trainvalidation true \
  --data trainvalidate_sirsam/   \
  --config nn_regression_keras.py   \
  --epochs 20 \
  --iterations 5


# extract query for prediction
landshark-extract query \
    --features features_sirsam.hdf5 \
    --strip 5 10 \
    --name sirsam \
    --halfwidth 1

# predict using train/test data based training
landshark --keras-model --batch-mb 0.001 predict \
    --config nn_regression_keras.py \
    --checkpoint nn_regression_keras_model_1of10 \
    --data query_sirsam_strip5of10

# predict using train/validation data based training
landshark --keras-model --batch-mb 0.001 predict \
    --config nn_regression_keras.py \
    --checkpoint nn_regression_keras_model_train_validation/ \
    --data query_sirsam_strip5of10


# oos validation block

# import oos data from oos shapefile into target oos hdf5 file
# note I am using the same shapefile even as oos shapefile here
landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name sirsam_oos \
  --record Na_ppm_i_1 \
  --dtype continuous


# extract oos targets into tfrecord
# TODO: can avoid saving one of train.xxx.tfrecord and test.xxx.tfrecord
landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
  --features features_sirsam.hdf5 \
  --split 1 1 \
  --targets targets_sirsam_oos.hdf5 \
  --name sirsam_oos \
  --halfwidth 1

# predict oos score
landshark -v DEBUG --keras-model --batch-mb 0.001 predict_oos \
    --proba false \
    --config nn_regression_keras.py \
    --checkpoint nn_regression_keras_model_1of10 \
    --data traintest_sirsam_oos_fold1of1 \
    --pred_ensemble_size 12000000  # not used