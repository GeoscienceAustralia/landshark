set -e
landshark-import --nworkers 0 --batch-mb 0.001 tifs \
    --name sirsam \
    --ignore-crs  \
    --continuous ../integration/data/continuous

landshark-import --batch-mb 0.001 targets   \
    --name groups_sirsam   --record group_cat   \
    --dtype continuous --group_col group \
    --shapefile ../integration/data/targets/geochem_sites_groups_15.shp;

for i in {1..5};
do
  echo fold $i of 5
  landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
    --features features_sirsam.hdf5 \
    --split ${i} 5 \
    --targets targets_groups_sirsam.hdf5 \
    --name sirsam \
    --halfwidth 1 --group_kfold;

  landshark --keras-model train \
    --trainvalidation false \
    --data traintest_sirsam_fold${i}of5/ \
    --config nn_regression_keras.py  \
    --epochs 20 \
    --iterations 5;
done;

