landshark-import --nworkers 2 --batch-mb 0.001 tifs --name sirsam --ignore-crs --continuous integration/data/continuous
landshark-import --nworkers 2 --batch-mb 0.001 targets --shapefile integration/data/targets/geochem_sites.shp --name Na_ppm_i_1 --record Na_ppm_i_1 --dtype continuous
landshark-extract --nworkers 2 --batch-mb 0.001 traintest --features features_sirsam.hdf5 --split 1 10 --targets targets_Na_ppm_i_1.hdf5 --name sirsam
landshark-extract --nworkers 2 --batch-mb 0.001 query --features features_sirsam.hdf5 --strip 5 10 --name sirsam
landshark train --data traintest_sirsam_fold1of10 --config configs/nn_regression.py --epochs 200 --iterations 5
landshark --batch-mb 0.001 predict --config configs/nn_regression.py --checkpoint nn_regression_model_1of10 --data query_sirsam_strip5of10
