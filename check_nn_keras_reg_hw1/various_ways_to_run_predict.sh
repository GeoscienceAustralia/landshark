# parallel query_predict ::: {1..5} ::: 5
#1 5
#2 5
#3 5
#4 5
#5 5

# gdal_merge.py -o merged.tif nn_regression_keras_model_1of10/predictions_Na_ppm_i_1_*of4.tif

# merge statement
# echo gdal_merge.py \\ > merge.txt
# for i in {1..384}; do echo nn_regression_keras_global_local_model_6of10/predictions_log_cond_1_${i}of384.tif >> merge.txt \\ ; done
# echo -o predictions_log_cond_1.tif >> merge.txt
# bash merge.txt
# convert to cogs
# -t_srs EPSG:3577 -of COG -co BIGTIFF=YES -co COMPRESS=LZW

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


# parallel -u --link  --dryrun echo {1} {2} ">" {1}.log ::: bird flower fist ::: red green blue
# parallel -u --link  echo {1} {2} ">" {1}.log ::: bird flower fist ::: red green blue

# post processing
# run with 10 nodes with 48 cores each
# parallel -u qsub {}_pred_parts.sh ::: {1..10}

# echo gdal_merge.py \\ > merge.txt
# for i in {1..480}; do echo nn_regression_keras_global_local_model_6of10/predictions_log_cond_1_${i}of480.tif >> merge.txt \\ ; done
# echo -o predictions_log_cond_1.tif >> merge.txt
# bash merge.txt

# sed -i 's/predictions_log_cond_12/predictions_cond_12/g' merge.txt