.PHONY: help clean clean-pyc clean-build typecheck typecheck-xml lint lint-xml test test-xml integration

help:
	@echo "clean - clean all artefacts"
	@echo "pred-clean - clean all helper prediction qsub files "
	@echo "query-clean - clean all helper qeery dirs"
	@echo "hdf-clean - clean all hdf5 files"
	@echo "model-clean - clean all model dirs"
	@echo "traintest-clean - clean all traintest dirs"

config = nn_regression_keras_global_local
name = sirsam
halfwidth = 1
epochs = 15
batchsize = 100
iterations = 10
fold = 2
total_folds = 5  # WARNING: if you change this, change parallel commands too
query_fold = 5
query_total_folds = 100  # WARNING: if you change this, change parallel commands too
data = ../integration/data/continuous
shape = ../integration/data/targets/geochem_sites_groups_15.shp
validation_shape = ../integration/data/targets/geochem_sites_groups_15.shp
predict_nodes = 10
prediction_ensemble_size = 5
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
run-five-fold-cross-val: import-features import-targets five-fold-cross-val

train-validation-model: import-features import-validation-targets extract-trainvalidation train-trainvalidation query predict-trainvalidation


oos-import-targets:
	landshark-import --batch-mb 0.001 targets \
	  --shapefile $(validation_shape) \
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
    --proba true \
    --config $(config).py \
    --checkpoint $(config)_model_$(fold)of$(total_folds) \
    --data traintest_${name}_oos_fold1of1 \
    --pred_ensemble_size $(prediction_ensemble_size)  # not used for proba=false


import-validation-targets:
	landshark-import --batch-mb 0.001 targets \
    --name validation_${name} \
    --shapefile $(shape) \
    --dtype continuous \
    --record Na_ppm_i_1 \
    --record Zr_ppm_i_1
    #    --group_col $(group_col) \

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
    --record Na_ppm_i_1 \
    --record Zr_ppm_i_1
    #    --group_col $(group_col) \


define extract_traintest_one_fold
	$(eval $@_fold = $(1))
    $(eval $@_total_folds = $(2))
    landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
           	--features features_${name}.hdf5 \
           	--split ${$@_fold} ${$@_total_folds} \
           	--targets targets_${name}.hdf5 \
         	--name ${name} \
         	--kfold \
           	--halfwidth ${halfwidth} \
			# --group_kfold
endef

extract-traintest:
	@$(call extract_traintest_one_fold, $(fold), $(total_folds))


define traintest_one_fold
	$(eval $@_fold = $(1))
    $(eval $@_total_folds = $(2))
    landshark --keras-model train \
	  --trainvalidation false \
	  --data traintest_${name}_fold${$@_fold}of${$@_total_folds} \
	  --config ${config}.py  \
	  --epochs ${epochs} \
	  --iterations ${iterations}
endef

run_traintest_demo:
	@$(call traintest_one_fold, 7, 10)

train-traintest:
	@$(call traintest_one_fold, $(fold), $(total_folds))

five-fold-cross-val:
	@echo ============================Will run 5 fold cross val in parallel==============================
	@echo ============================Will run 5 fold cross val in parallel==============================
	@echo ============================Will run 5 fold cross val in parallel==============================
	@make -j train-fold1 train-fold2 train-fold3 train-fold4 train-fold5
	@echo ============================5-fold cross-validation finished ==================================
	@echo ============================5-fold cross-validation finished ==================================
	@echo ============================5-fold cross-validation finished ==================================



train-fold1:
	@$(call extract_traintest_one_fold, 1, 5)
	@$(call traintest_one_fold, 1, 5)

train-fold2:
	@$(call extract_traintest_one_fold, 2, 5)
	@$(call traintest_one_fold, 2, 5)

train-fold3:
	@$(call extract_traintest_one_fold, 3, 5)
	@$(call traintest_one_fold, 3, 5)

train-fold4:
	@$(call extract_traintest_one_fold, 4, 5)
	@$(call traintest_one_fold, 4, 5)

train-fold5:
	@$(call extract_traintest_one_fold, 5, 5)
	@$(call traintest_one_fold, 5, 5)

train-final-model:
	python submit_with_optimal_epochs.py \
		-c $(config).py \
		-n $(name) \
		-b $(batchsize) \
		-e $(epochs) \
		-f $(total_folds) \
		-i $(iterations) \
		-w $(halfwidth)

#@$(call extract_traintest_one_fold, 1, 1)
#@$(call traintest_one_fold, 1, 1)

# @$(call traintest_one_fold, $(fold), $(total_folds))
#parallel -u --dryrun @$(call traintest_one_fold, $(fold), $(total_folds)) {} ::: 1
# parallel -u --dryrun @$(call traintest_one_fold, $(fold), $(total_folds)) ::: 1

query:
	landshark-extract query \
    --features features_${name}.hdf5 \
    --strip $(query_fold) $(query_total_folds) \
    --name ${name} \
    --halfwidth $(halfwidth)

predict-traintest:
	landshark --keras-model --batch-mb 0.001 predict \
	--proba true \
	--config $(config).py \
	--checkpoint ${config}_model_$(fold)of$(total_folds) \
    --data query_${name}_strip$(query_fold)of$(query_total_folds) \
    --pred_ensemble_size $(prediction_ensemble_size)

predict-trainvalidation:
	landshark --keras-model --batch-mb 0.001 predict \
	--proba true \
	--config $(config).py \
	--checkpoint ${config}_model_train_validation \
    --data query_${name}_strip$(fold)of$(total_folds)


#query_predict: query predict
#
#predict_proba:
#	python predict_automation.py ${predict_nodes}
#	for i in `seq 1 ${predict_nodes}` ; do qsub $${i}_$${pred_parts}; done
