import os
import argparse
import tensorflow as tf
import numpy as np
import mlflow
# import mlflow.tensorflow
from tensorflow.examples.tutorials.mnist import input_data

pjoin = os.path.join


def artificial_neural_network(
    X: tf.placeholder,
    y: tf.placeholder,
    n_hidden: int,
    hidden_dim: int,
    n_outputs: int,
    lr: float,
):
    """Build an artificial neural network with a gradient descent optimizer
    
    :params X:
    :params y:
    :params n_hidden:
    :params lr:
    """
    with tf.variable_scope("dnn_model", reuse=tf.AUTO_REUSE):
        hidden = X
        for i in range(n_hidden):
            hidden = tf.layers.dense(hidden, hidden_dim, name=f"h{i}", activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, n_outputs, name="logits")
    
    with tf.variable_scope("metrics", reuse=tf.AUTO_REUSE):
        xentropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        data_cost = tf.reduce_mean(xentropy_loss, name="data_loss")
        top1 = tf.nn.in_top_k(logits, y, 1)
        top3 = tf.nn.in_top_k(logits, y, 3)

        accuracy = tf.reduce_mean(tf.cast(top1, tf.float32))
        top3_accuracy = tf.reduce_mean(tf.cast(top3, tf.float32))

        y_proba = tf.nn.softmax(logits, name="softmax")
        
    with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
        global_step = tf.get_variable(name="global_step", dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        training_op = optimizer.minimize(data_cost, global_step=global_step)
        
    with tf.variable_scope("save", reuse=tf.AUTO_REUSE):
        saver = tf.train.Saver(name="saver")
        
    with tf.variable_scope("summaries", reuse=tf.AUTO_REUSE):   
        tf.summary.scalar("data_cost_summary", data_cost)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("top3_accuracy", top3_accuracy)

    merged = tf.summary.merge_all()
    
    return saver, training_op, y_proba, data_cost, merged


# mlflow.tensorflow.autolog()
def train_fashion_mnist_ann(
    fashion_mnist,
    image_width,
    image_height,
    n_epochs,
    learning_rate,
    batch_size,
    n_classes,
    n_hidden,
    hidden_dim,
    checkpoint_dir,
):
    """
    :param fashion_mnist: 
    :param image_width:
    :param image_height:
    :param n_epochs: full passes through the training data
    :param learning_rate: gradient multiplier in SGD
    :param batch_size: number of examples to use in SGD
    :param n_classes: number of class labels
    :param n_hidden: number of hidden layers
    :param hidden_dim: number of hidden units in a layer
    :param checkpoint_dir: directory to checkpoint the model
    """
    with mlflow.start_run():
        run_uuid = mlflow.active_run().info.run_uuid
        print(f"-" * 50, flush=True)
        print(f"MLflow Run ID: {run_uuid}", flush=True)
        print(f"-" * 50, flush=True)

        mlflow.log_params(
            {
                "n_epochs": n_epochs,
                "learning rate": learning_rate,
                "batch_size": batch_size,
                "n_hidden": n_hidden,
                "hidden_dim": hidden_dim,
            }
        )
        tf.reset_default_graph()

        train_dir = pjoin("tf_logs", "train", run_uuid)
        validation_dir = pjoin("tf_logs", "test", run_uuid)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        log_every = 1
        n_train = fashion_mnist.train.num_examples
        # n_validate = fashion_mnist.validation.num_examples

        X = tf.placeholder(tf.float32, shape=(None, image_width * image_height), name="X")
        y = tf.placeholder(tf.int64, shape=(None,), name="y")
        n_train_batches, remainder = divmod(n_train, batch_size)
        if remainder > 0:
            n_train_batches += 1

        saver, training_op, _, data_cost, merged = artificial_neural_network(
            X,
            y,
            n_hidden,
            hidden_dim,
            n_classes,
            learning_rate
        )
        train_writer = tf.summary.FileWriter(train_dir, tf.get_default_graph())
        test_writer = tf.summary.FileWriter(validation_dir, tf.get_default_graph())

        with tf.Session() as sess:
            # try:
            #     saver.restore(sess, checkpoint_dir)
            # except Exception:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable("global_step", dtype=tf.int32)

            for i in range(n_epochs):
                for _ in range(n_train_batches):
                    X_train_batch, y_train_batch = fashion_mnist.train.next_batch(batch_size)
                    _, train_cost, train_summ = sess.run(
                        [training_op, data_cost, merged], 
                        feed_dict={
                            X: X_train_batch, 
                            y: y_train_batch},
                    )

                if i % log_every == 0:
                    nodes_to_eval = [merged, data_cost, global_step]
                    data_to_feed = {X: fashion_mnist.validation.images, y: fashion_mnist.validation.labels}
                    validation_summ, validation_loss, g_step = sess.run(
                        nodes_to_eval,
                        feed_dict=data_to_feed,
                    )
                    epoch = (g_step + 1) // n_train_batches
                    mlflow.log_metrics({
                        "validation_loss": validation_loss,
                        "train_loss": train_cost,
                    }, step=epoch)
                    train_writer.add_summary(train_summ, epoch)
                    test_writer.add_summary(validation_summ, epoch)
                    # if checkpoint_dir is not None:
                    #     save_path = saver.save(sess, checkpoint_dir)
            # if checkpoint_dir:
            #     save_path = saver.save(sess, checkpoint_dir)
        train_writer.close()
        test_writer.close()

        mlflow.log_artifacts(train_dir)
        mlflow.log_artifacts(validation_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-width", type=int, default=28)
    parser.add_argument("--image-height", type=int, default=28)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--checkpoint-dir", default=None)

    args = parser.parse_args()
    fashion_mnist_data_gen = input_data.read_data_sets(
        pjoin("data", "fashion"),
        source_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    )

    train_fashion_mnist_ann(
        fashion_mnist_data_gen,
        args.image_width,
        args.image_height,
        args.n_epochs,
        args.learning_rate,
        args.batch_size,
        args.n_classes,
        args.n_hidden,
        args.hidden_dim,
        args.checkpoint_dir,
    )
