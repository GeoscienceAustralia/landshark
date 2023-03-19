landshark-import --nworkers 0 --batch-mb 0.001 tifs \
    --name sirsam \
    --ignore-crs  \
    --continuous ../integration/data/continuous

landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name Na_ppm_i_1 \
  --record Na_ppm_i_1 \
  --record Zr_ppm_i_1 \
  --dtype continuous

landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
  --features features_sirsam.hdf5 \
  --split 1 10 \
  --targets targets_Na_ppm_i_1.hdf5 \
  --name sirsam \
  --halfwidth 0

landshark --keras-model train \
  --data traintest_sirsam_fold1of10 \
  --config nn_regression_keras.py \
  --epochs 20 \
  --iterations 50

landshark-extract query \
    --features features_sirsam.hdf5 \
    --strip 5 10 \
    --name sirsam \
    --halfwidth 0

landshark --keras-model --batch-mb 0.001 predict \
    --config nn_regression_keras.py \
    --checkpoint nn_regression_keras_model_1of10 \
    --data query_sirsam_strip5of10