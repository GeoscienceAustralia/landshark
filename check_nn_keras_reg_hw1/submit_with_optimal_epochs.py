import math
from subprocess import run
from optparse import OptionParser
from pathlib import Path
import pandas as pd


def method_name(name, total_folds, config_name, halfwidth):
    num_interations = []
    val_fold_r2s = []
    config_fname = Path(config_name).stem
    for i in range(1, total_folds + 1, 1):
        c = Path(config_fname + "_model_" + f"{i}of{total_folds}").joinpath("training_scores.csv")
        print(c)
        df = pd.read_csv(c)
        num_interations.append(int(df.loc[df['val_loss'].idxmin()].epoch) + 1)
        val_fold_r2s.append(float(df.loc[df['val_loss'].idxmin()].val_r2))
    print(num_interations)
    print(val_fold_r2s)
    print(f"average number of iterations: {math.ceil(sum(num_interations) / total_folds)}")
    print(f"average group kold performance: {sum(val_fold_r2s) / total_folds}")
    avg_num_iterations = math.ceil(sum(num_interations) / total_folds)
    run(
        f"landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
        --features ./features_{name}.hdf5 \
        --split 1 1 \
        --targets ./targets_{name}.hdf5 \
        --name {name} \
        --halfwidth {halfwidth};", shell=True
    )
    run(
        f"landshark --keras-model train \
        --trainvalidation false \
        --data ./traintest_{name}_fold1of1 \
        --config {config_name}  \
        --epochs {epochs} \
        --iterations {avg_num_iterations} \
        --batchsize {batchsize};", shell=True
    )


if __name__ == '__main__':
    epochs = 10
    batchsize = 100
    iterations = 500
    total_folds = 5
    config = "./nn_regression_keras.py"
    name = "sirsam"

    parser = OptionParser(usage='%prog -c config_file_name \n'
                                '-h halfwidth')
    parser.add_option('-c', '--config', type=str, dest='config',
                      help='name of python config file')

    parser.add_option('-h', '--halfwidth', type=float, dest='halfwidth',
                      default=0,
                      help='Halfwidth of the patch on \n'
                           'either side of the target')
    options, args = parser.parse_args()
    options.name = name
    options.total_folds = total_folds

    if not options.config:  # if filename is not given
        parser.error('Provide config file')

    if not options.halfwidth:  # if filename is not given
        parser.error('Provide halfwidth')

    method_name(options.name, options.total_folds, options.config, options.halfwidth)
