set -e

# import features
#landshark-import --nworkers 0 --batch-mb 0.001 tifs \
#    --name sirsam \
#    --ignore-crs  \
#    --continuous ../integration/data/continuous
#
## import targets
#landshark-import --batch-mb 0.001 targets   \
#    --name groups_sirsam   --record group_cat   \
#    --dtype continuous --group_col group \
#    --shapefile ../integration/data/targets/geochem_sites_groups_15.shp;

function train_fold {
  i=$1
  epochs=15
  batchsize=100
  iterations=500
  total_folds=5
  echo extract and train fold ${i} of ${total_folds};
  # extract fold
  landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
    --features features_sirsam.hdf5 \
    --split ${i} ${total_folds} \
    --targets targets_groups_sirsam.hdf5 \
    --name sirsam \
    --halfwidth 1 --group_kfold;

  # train
  landshark --keras-model train \
    --trainvalidation false \
    --data traintest_sirsam_fold${i}of${total_folds}/ \
    --config nn_regression_keras.py  \
    --epochs ${epochs} \
    --iterations ${iterations} \
    --batchsize ${batchsize};
}

export -f train_fold

#parallel train_fold ::: {1..5}
#parallel -u --progress train_fold {} ">" {}.log  ::: {1..5}

# ::: ${total_folds} ::: 500 ::: 100

echo Now running optimal number of iterations with average number of partitions
python submit_with_optimal_epochs.py

#for i in {1..5};
#do
#  echo fold ${i} of 5
#  train_fold ${i}
#done;

