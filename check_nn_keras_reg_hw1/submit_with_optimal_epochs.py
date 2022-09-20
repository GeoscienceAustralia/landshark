import math
from subprocess import run
from pathlib import Path

epochs = 15
batchsize = 100
iterations = 500
total_folds = 5

num_interations = []
val_fold_r2s = []

for i in range(1, total_folds + 1, 1):
    c = Path("nn_regression_keras_model_" + f"{i}of{total_folds}").joinpath("training_scores.csv")
    print(c)
    with open(c, 'r') as readder:
        final_line = readder.readlines()[-1].strip("\n")
        num_interations.append(int(final_line.split(',')[0]))
        val_fold_r2s.append(float(final_line.split(',')[-1]))

print(num_interations)
print(val_fold_r2s)
print(f"average number of iterations: {math.ceil(sum(num_interations) / total_folds)}")
print(f"average group kold performance: {sum(val_fold_r2s) / total_folds}")
avg_num_iterations = math.ceil(sum(num_interations) / total_folds)

run(
    "landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
    --features features_sirsam.hdf5 \
    --split 1 1 \
    --targets targets_groups_sirsam.hdf5 \
    --name sirsam \
    --halfwidth 1;", shell=True
)

run(
    f"landshark --keras-model train \
    --trainvalidation false \
    --data traintest_sirsam_fold1of1/ \
    --config nn_regression_keras.py  \
    --epochs {epochs} \
    --iterations {avg_num_iterations} \
    --batchsize {batchsize};", shell=True
)
