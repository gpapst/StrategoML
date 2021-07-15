import csv
import tensorflow as tf
# import multiprocessing as mp

from Training.generator import WDataGenerator
from q_learning.qnet import QResNet

PATH = "E:\\games"
TESTPATH = "E:\\test_games"


def scheduler(epoch, lr):
    """
    learning rate schedule (placeholder)
    :param epoch:
    :param lr:
    :return:new Learning rate
    """
    return lr * 0.9


if __name__ == "__main__":
    net = QResNet()

    net.Model.summary()
    print(net.Model.metrics)

    checkpoint_path = "folder to save checkpoints"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=False,
                                                     save_weights_only=False, save_frequency=1)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    data_generator = WDataGenerator(1024, 950, fpath=PATH)
    eval_generator = WDataGenerator(1024, 90, fpath=TESTPATH)

    current_ep = 0
    eps = 10
    print('Starting...')
    while current_ep < eps:
        training = net.Model.fit(x=data_generator, steps_per_epoch=900, epochs=current_ep + 1,
                                 validation_data=eval_generator, validation_steps=80, workers=6,
                                 use_multiprocessing=True, initial_epoch=current_ep,
                                 callbacks=[lr_callback, cp_callback])

        print(training.history)
        net.Model.save_weights(filepath="E:\\QRES\\weights.h5", save_format='h5')
        with open(r"E:\QRES.csv", "a") as file:
            writer = csv.DictWriter(file, fieldnames=[key for key in training.history.keys()], extrasaction='ignore')
            if current_ep == 0:
                writer.writeheader()
            writer.writerow(training.history)
        current_ep += 1

""""""
