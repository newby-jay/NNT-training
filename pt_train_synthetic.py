from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import os.path
import time
import pylab as pl
import numpy as np
import tensorflow as tf
from scipy import rand

import pt

from pt_input_synthetic import makeVid

imageSize = 256
learningRate = 0.085
decayFactor = 0.8
savePath = 'gs://ntdev/Net_RNN-2p/'


def normalize(phi):
    e1 = np.exp(phi[0, ..., 1] - phi[0, ..., 0])
    Z = e1 + 1.
    return e1/Z

def train(n):
    train_dir = os.path.join(savePath, 'set{0}'.format(int(n)))
    if not tf.gfile.Exists(train_dir):
        tf.gfile.MkDir(train_dir)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = makeVid()

        logits = pt.processFrames(images, Nframes=5, dprob=0.53)
        loss = pt.loss(logits, labels)
        train_op, lr = pt.train(loss, global_step, learningRate, decayFactor)


        saver = tf.train.Saver(max_to_keep=None)
        sess = tf.InteractiveSession(config=tf.ConfigProto(
            log_device_placement=False))

        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        max_steps = 300000
        lastTime = 0
        max_steps -= lastTime
        for step in np.arange(lastTime, max_steps+lastTime):
            start_time = time.time()
            _, loss_value, prediction, lrj = sess.run(
                [train_op, loss, logits, lr])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 40 == 0 or step == 1:
                sec_per_batch = float(duration)
                P1 = normalize(prediction)
                print('%s: step %d, lr = %.4f, loss = %.6f, Pminmax = %.3f/%.3f, (%.3f sec/batch)'
                       % (datetime.now(), step, lrj, loss_value, P1.min(), P1.max(), sec_per_batch))
            if step % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        sess.close()
if __name__ == '__main__':
    train(1)
