.PHONY: help clean clean-pyc clean-build typecheck typecheck-xml lint lint-xml test test-xml integration

help:
	@echo "clean - clean all artefacts"

config = nn_regression_keras_global_local
name = sirsam
halfwidth = 1
epochs = 15
batchsize = 100
iterations = 5
total_folds = 5  # WARNING: if you change this, change parallel commands too
data = ../integration/data/continuous
shape = ../integration/data/targets/geochem_sites_groups_15.shp


clean: clean-traintest clean-model clean-hdf clean-query clean-pred

clean-traintest:
	rm -fr traintest_$(name)_fold*

clean-model:
	rm -fr $(config)_model_*/

clean-query:
	rm -fr query_$(name)_strip*/

clean-hdf:
	rm -fr features_$(name).hdf5 targets_$(name).hdf5

clean-pred:
	rm -f *_pred_parts.sh

import-features:
	landshark-import --nworkers 0 --batch-mb 0.001 tifs \
        --name ${name} \
        --ignore-crs  \
        --continuous $(data)

import-targets:
	landshark-import --batch-mb 0.001 targets \
    --name ${name} --record group_cat \
    --dtype continuous --group_col group \
    --shapefile ${shape}

#landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
#  --features features_sirsam.hdf5 \
#  --split 1 10 \
#  --targets targets_Na_ppm_i_1.hdf5 \
#  --name sirsam \
#  --halfwidth 0
#
#landshark --keras-model train \
#  --trainvalidation false \
#  --data traintest_sirsam_fold1of10/ \
#  --config nn_regression_keras_global_local.py  \
#  --epochs 20 \
#  --iterations 100