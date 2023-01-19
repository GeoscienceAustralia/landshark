set -e
export name="sirsam"
export halfwidth=1
export config="./nn_regression_keras.py"
export epochs=15
export batchsize=100
export iterations=5
export total_folds=5  # WARNING: if you change this, change parallel commands too
export config_basename=${config##*/}
export config_stem=${config_basename%.*}
echo $config  $config_basename $config_stem


# import features
landshark-import --nworkers 0 --batch-mb 0.001 tifs \
    --name ${name} \
    --ignore-crs  \
    --continuous ../integration/data/continuous

# import targets
landshark-import --batch-mb 0.001 targets \
    --name ${name} --record group_cat \
    --dtype continuous --group_col group \
    --shapefile ../integration/data/targets/geochem_sites_groups_15.shp;

function train_fold {
  i=$1
  total_folds=$2
  echo extract and train fold ${i} of ${total_folds};
  # extract fold
  landshark-extract --nworkers 10 --batch-mb 0.1 traintest \
    --features ./features_${name}.hdf5 \
    --targets ./targets_${name}.hdf5 \
    --name $name --halfwidth ${halfwidth} \
    --split "${i}" "${total_folds}" --group_kfold;

  landshark --keras-model --no-gpu train \
    --config ${config} --epochs ${epochs} --iterations ${iterations} \
    --data ./traintest_${name}_fold${i}of${total_folds}/ \
    --batchsize ${batchsize};
}

export -f train_fold

parallel  mkdir -p ${config_stem}_model_{1}of{2}/ ::: {1..5} ::: 5
parallel -u --progress train_fold {1} {2} ">" \
  ${config_stem}_model_{1}of{2}/${PBS_JOBNAME}_${PBS_JOBID}_{1}.log ::: {1..5} ::: 5

python submit_with_optimal_epochs.py \
  -c ${config} -w ${halfwidth} -b $batchsize -i ${iterations} \
  -f $total_folds -e $epochs -n $name


#function query_predict {
#    echo starting query and preict $1 of $2
#    n=$1
#    N=$2
#    mkdir -p query_${name}_strip${1}of${2}
#    landshark-extract --nworkers 0 --batch-mb 0.1 query \
#        --features ./features_${name}.hdf5 \
#        --strip ${n} ${N} --name ${name} \
#        --halfwidth ${halfwidth}
#    landshark --keras-model --batch-mb 0.1 predict \
#        --config ${config} \
#        --checkpoint ${config_stem}_model_1of1 \
#        --data query_${name}_strip${n}of${N}
#    echo done query and preict $1 of $2
#}
#export -f query_predict
#
#parallel  mkdir -p query_${name}_strip{1}of{2} ::: {1..8} ::: 8
#parallel -u -j 4 query_predict {1} {2} ">" query_${name}_strip{1}of{2}/${PBS_JOBNAME}_${PBS_JOBID}_{1}.log ::: {1..8} ::: 8
exit 0
