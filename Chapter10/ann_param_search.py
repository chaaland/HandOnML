import argparse
import os
import pickle as pkl
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from tensorflow.examples.tutorials.mnist import input_data
from ann_training import fit_fashion_mnist_ann

pjoin = os.path.join


def main(args):
    max_evals = args.n_trials

    try:
        trials = pkl.load(open(args.hyperopt_trials, "rb"))
        print(f"Loaded saved HyperOpt Trials object from {args.hyperopt_trials}", flush=True)
        n_prev_trials = len(trials.trials)
        max_evals += n_prev_trials
        print(f"Rerunning from {n_prev_trials} trials.", flush=True)
    except:
        trials = Trials()
        print("No saved HyperOpt Trials object found. Starting from scratch", flush=True)

    space = {
        "image_width": 28,
        "image_height": 28,
        "n_classes": 10,
        "n_epochs": hp.choice("n_epochs", range(20,50)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.00001), np.log(0.1)),
        "batch_size": hp.choice("batch_size", [2**i for i in range(5,10)]),
        "n_hidden": hp.choice("n_hidden", range(1,4)),
        "hidden_dim": hp.choice("hidden_dim", [2**i for i in range(5,10)]),
    }

    fashion_mnist_data_gen = input_data.read_data_sets(
        pjoin("data", "fashion"),
        source_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    )

    best = fmin(
        fn=lambda hps: fit_fashion_mnist_ann(fashion_mnist_data_gen, **hps),
        space=space,
        algo=tpe.suggest, 
        max_evals=max_evals,
        trials=trials,
    )

    pkl.dump(trials, open(args.hyperopt_trials, "wb"))

    best_acc = max(-trial_data["result"]["loss"] for trial_data in trials.trials)
    print(f"Best validation accuracy: {100 * best_acc:.4}%", flush=True)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--hyperopt-trials", type=str, default="hyper_opt_trials.pkl")
    args = parser.parse_args()

    main(args)
