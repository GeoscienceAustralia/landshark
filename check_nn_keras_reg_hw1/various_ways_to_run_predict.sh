# parallel query_predict ::: {1..5} ::: 5
#1 5
#2 5
#3 5
#4 5
#5 5

# gdal_merge.py -o merged.tif nn_regression_keras_model_1of10/predictions_Na_ppm_i_1_*of4.tif

# merge statement
# echo gdal_merge.py -o predictions_log_cond_1.tif \\ > merge.txt
# for i in {1..384}; do echo nn_regression_keras_global_local_model_6of10/predictions_log_cond_1_${i}of384.tif >> merge.txt \\ ; done
# bash merge.txt

#for n in {1..N};
#  do echo $n;
#
#done
#
#landshark-extract --nworkers 0 --batch-mb 0.001 query \
#  --features features_sirsam.hdf5 \
#  --strip 5 10 --name sirsam \
#  --halfwidth 1
#
#landshark --keras-model --batch-mb 0.001 predict \
#  --config nn_regression_keras.py \
#  --checkpoint nn_regression_keras_model_1of10 \
#  --data query_sirsam_strip5of10
