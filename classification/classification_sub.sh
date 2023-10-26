echo Importing features...
landshark-import --nworkers 0 --batch-mb 0.001 tifs --name sirsam --ignore-crs --categorical /home/sudipta/repos/landshark/integration/data/categorical --continuous /home/sudipta/repos/landshark/integration/data/continuous
echo Importing targets...
landshark-import --batch-mb 0.001 targets --shapefile /home/sudipta/repos/landshark/integration/data/targets/geochem_sites.shp --name SAMPLETYPE --record SAMPLETYPE --dtype categorical

echo Extracting training data...
landshark-extract --nworkers 0 --batch-mb 0.001 traintest --features features_sirsam.hdf5 --split 1 10 --targets targets_SAMPLETYPE.hdf5 --name sirsam --halfwidth 0

echo Training...
landshark --keras-model train --data traintest_sirsam_fold1of10 --config /home/sudipta/repos/landshark/configs/nn_classification_keras.py --epochs 200 --iterations 5

echo Extracting query data...
landshark-extract --nworkers 0 --batch-mb 0.001 query --features features_sirsam.hdf5 --strip 5 10 --name sirsam --halfwidth 0
echo Predicting...
landshark --keras-model --batch-mb 0.001 predict --config nn_classification_keras.py --checkpoint nn_classification_keras_model_1of10 --data query_sirsam_strip5of10
