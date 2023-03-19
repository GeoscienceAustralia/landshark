.PHONY: help clean clean-pyc clean-build typecheck typecheck-xml lint lint-xml test test-xml integration

help:
	@echo "clean - clean all artefacts"
	@echo "pred-clean - clean all helper prediction qsub files "
	@echo "query-clean - clean all helper qeery dirs"
	@echo "hdf-clean - clean all hdf5 files"
	@echo "model-clean - clean all model dirs"
	@echo "traintest-clean - clean all traintest dirs"

config = nn_regression_keras
name = sirsam
halfwidth = 1
epochs = 15
batchsize = 100
iterations = 10
fold = 5
total_folds = 10  # WARNING: if you change this, change parallel commands too
data = ../integration/data/continuous
shape = ../integration/data/targets/geochem_sites_groups_15.shp
validation_shape = ../integration/data/targets/geochem_sites_groups_15.shp
predict_nodes = 10
group_col = group
pred_parts = pred_parts.sh


clean: traintest-clean model-clean hdf-clean query-clean pred-clean

traintest-clean:
	rm -fr traintest_$(name)_fold*
	rm -fr traintest_$(name)_oos_fold*
	rm -fr trainvalidate_$(name)


model-clean:
	rm -fr $(config)_model_*/

query-clean:
	rm -fr query_$(name)_strip*/

hdf-clean:
	rm -fr features_$(name).hdf5 targets_$(name).hdf5 targets_$(name)_oos.hdf5  targets_validation_$(name).hdf5

pred-clean:
	rm -f *_pred_parts.sh
	rm -f *_pred_parts.sh.o*

oos-validation: train-test-model oos-import-targets oos-extract oos-predict
train-test-predict-model: train-test-model query predict-traintest
train-test-model: import-features import-targets extract-traintest train-traintest

train-validation-model: import-features import-validation-targets extract-trainvalidation train-trainvalidation query predict-trainvalidation


oos-import-targets:
	landshark-import --batch-mb 0.001 targets \
	  --shapefile $(shape) \
	  --name ${name}_oos \
	  --record Na_ppm_i_1 \
	  --record Zr_ppm_i_1 \
	  --dtype continuous

oos-extract:
	landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
	--features features_${name}.hdf5 \
	--split 1 1 \
	--targets targets_${name}_oos.hdf5 \
	--name ${name}_oos \
	--halfwidth $(halfwidth)

# predict oos score
oos-predict:
	landshark -v DEBUG --keras-model --batch-mb 0.001 predict_oos \
    --proba false \
    --config $(config).py \
    --checkpoint $(config)_model_$(fold)of$(total_folds) \
    --data traintest_${name}_oos_fold1of1 \
    --pred_ensemble_size 12000000  # not used for proba=false


import-validation-targets:
	landshark-import --batch-mb 0.001 targets \
    --name validation_${name} \
    --shapefile $(shape) \
    --dtype continuous \
    --group_col $(group_col) \
    --record Na_ppm_i_1 \
    --record Zr_ppm_i_1

extract-trainvalidation:
	landshark-extract --nworkers 0 --batch-mb 0.01 trainvalidate \
  	--features features_${name}.hdf5 \
  	--targets targets_validation_${name}.hdf5 \
  	--validation targets_validation_${name}.hdf5 \
  	--name ${name} \
  	--halfwidth ${halfwidth}

train-trainvalidation:
	landshark --keras-model train \
      --trainvalidation true \
      --data trainvalidate_${name}/   \
      --config ${config}.py   \
      --epochs 20 \
      --iterations 5


import-features:
	landshark-import --nworkers 0 --batch-mb 0.001 tifs \
	--name ${name} \
	--ignore-crs  \
	--continuous $(data)

import-targets:
	landshark-import --batch-mb 0.001 targets \
    --name ${name} \
    --shapefile $(shape) \
    --dtype continuous \
    --group_col $(group_col) \
    --record Na_ppm_i_1 \
    --record Zr_ppm_i_1

extract-traintest:
	landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
  	--features features_${name}.hdf5 \
  	--split $(fold) $(total_folds) \
  	--targets targets_${name}.hdf5 \
	--name ${name} \
	--group_kfold \
  	--halfwidth ${halfwidth}


train-traintest:
	landshark --keras-model train \
  --trainvalidation false \
  --data traintest_${name}_fold$(fold)of$(total_folds) \
  --config ${config}.py  \
  --epochs ${epochs} \
  --iterations ${iterations}


query:
	landshark-extract query \
    --features features_${name}.hdf5 \
    --strip $(fold) $(total_folds) \
    --name ${name} \
    --halfwidth $(halfwidth)

predict-traintest:
	landshark --keras-model --batch-mb 0.001 predict \
	--config $(config).py \
	--checkpoint ${config}_model_$(fold)of$(total_folds) \
    --data query_${name}_strip$(query_fold)of$(query_total_folds) \
    --pred_ensemble_size 5  # not used

predict-trainvalidation:
	landshark --keras-model --batch-mb 0.001 predict \
	--config $(config).py \
	--checkpoint ${config}_model_train_validation \
    --data query_${name}_strip$(fold)of$(total_folds)


#query_predict: query predict
#
#predict_proba:
#	python predict_automation.py ${predict_nodes}
#	for i in `seq 1 ${predict_nodes}` ; do qsub $${i}_$${pred_parts}; done
