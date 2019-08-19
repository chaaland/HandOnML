import os
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from tensorflow.examples.tutorials.mnist import input_data
from ann_training import fit_fashion_mnist_ann
import functools

pjoin = os.path.join


fashion_mnist_data_gen = input_data.read_data_sets(
        pjoin("data", "fashion"),
        source_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    )

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

trials = Trials()
best = fmin(
    fn=lambda hps: fit_fashion_mnist_ann(fashion_mnist_data_gen, **hps),
    # fn=functools.partial(
    #     fit_fashion_mnist_ann,
    #     fashion_mnist_data_gen,
    # ),
    space=space, 
    algo=tpe.suggest, 
    max_evals=5, 
    trials=trials,
)