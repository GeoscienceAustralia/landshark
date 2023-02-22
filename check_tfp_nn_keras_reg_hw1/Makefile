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
total_folds = 5  # WARNING: if you change this, change parallel commands too
data = ../integration/data/continuous
shape = ../integration/data/targets/geochem_sites_groups_15.shp
predict_nodes = 10
group_col = group
pred_parts = pred_parts.sh


clean: traintest-clean model-clean hdf-clean query-clean pred-clean

traintest-clean:
	rm -fr traintest_$(name)_fold*

model-clean:
	rm -fr $(config)_model_*/

query-clean:
	rm -fr query_$(name)_strip*/

hdf-clean:
	rm -fr features_$(name).hdf5 targets_$(name).hdf5

pred-clean:
	rm -f *_pred_parts.sh
	rm -f *_pred_parts.sh.o*


import-features:
	landshark-import --nworkers 0 --batch-mb 0.001 tifs \
        --name ${name} \
        --ignore-crs  \
        --continuous $(data)

import-targets:
	landshark-import --batch-mb 0.001 targets \
    --name ${name} \
    --dtype continuous --group_col $(group) \
    --shapefile ${shape} \
    --record group_cat

extract-traintest:
	landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
  	--features features_${name}.hdf5 \
  	--split $(fold) $(total_folds) \
  	--targets targets_${name}.hdf5 \
	--name ${name} \
  	--halfwidth ${halfwidth}

train:
	landshark --keras-model train \
  --trainvalidation false \
  --data traintest_${name}_fold$(fold)of$(total_folds)/ \
  --config ${config}.py  \
  --epochs ${epochs} \
  --iterations ${iterations}

query:
	bash pred_parts.sh query $(fold) $(total_folds)

predict:
	bash pred_parts.sh predict $(fold) $(total_folds)

query_predict: query predict

predict_proba:
	python predict_automation.py ${predict_nodes}
	for i in `seq 1 ${predict_nodes}` ; do qsub $${i}_$${pred_parts}; done
