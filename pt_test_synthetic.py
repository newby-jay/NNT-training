from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
from pylab import *
import numpy as np
import tensorflow as tf
from scipy import rand

import pt
from pt_input_synthetic import makeVid

imageSize = 256
savePath = 'gs://ntdev/Net_RNN-2p/'

def clipVid(I, c=0.99):
    I -= I.min()
    I /= I.max()
    h, Ibins = histogram(I.flatten(), 500)
    H = np.cumsum(h/sum(1.*h))
    Imin = Ibins[H<1.-c][-1] if sum(H<1.-c) > 0 else 0.
    Imax = Ibins[H>c][0] if sum(H>c) > 0 else 1.
    I = I.clip(Imin, Imax)
    return I

def normalize(phi):
    e1 = np.exp(phi[ ..., 1] - phi[ ..., 0])
    Z = e1 + 1.
    return e1/Z

def test(n):
    train_dir = os.path.join(savePath, 'set{0}'.format(int(n)))
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images, labels = makeVid()
        # labels = tf.zeros((3, 256, 256, 2))
        # images = tf.random_normal((1, 256, 256, 3))

        logits = pt.processFrames(images, Nframes=3, dprob=1., tflag=False)

        # summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver(max_to_keep=None)
        sess = tf.InteractiveSession(config=tf.ConfigProto(
            log_device_placement=False))

        init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        max_steps = 24
        p1, p2, p3 = 0., 0., 0.
        n1, n2, n3 = 0, 0, 0
        for step in np.arange(max_steps):
            start_time = time.time()
            prediction, I, L = sess.run([logits, images, labels])
            print('frames per second = %.3f'%(3./(time.time() - start_time)))

            fig = figure(1, [12, 12])

            P1 = normalize(prediction[0])
            p1 += sum(P1)
            n1 += sum(P1>0.5)
            P1[P1>0.5] = nan
            fig.add_subplot(331)
            imshow(P1, interpolation='none', vmin=0., vmax=1.)
            xticks([]);yticks([])
            fig.add_subplot(332)
            imshow(L[0, ..., 1], interpolation='none')
            xticks([]);yticks([])
            fig.add_subplot(333)
            imshow(I[0, ..., 0].copy(), interpolation='none', cmap='bone')
            xticks([]);yticks([])

            P1 = normalize(prediction[1])
            p2 += sum(P1)
            n2 += sum(P1>0.5)
            P1[P1>0.5] = nan
            fig.add_subplot(334)
            imshow(P1, interpolation='none', vmin=0., vmax=1.)
            xticks([]);yticks([])
            fig.add_subplot(335)
            imshow(L[1, ..., 1], interpolation='none')
            xticks([]);yticks([])
            fig.add_subplot(336)
            imshow(I[0, ..., 1].copy(), interpolation='none', cmap='bone')
            xticks([]);yticks([])

            P1 = normalize(prediction[2])
            p3 += sum(P1)
            n3 += sum(P1>0.5)
            P1[P1>0.5] = nan
            fig.add_subplot(337)
            imshow(P1, interpolation='none', vmin=0., vmax=1.)
            xticks([]);yticks([])
            fig.add_subplot(338)
            imshow(L[2, ..., 1], interpolation='none')
            xticks([]);yticks([])
            fig.add_subplot(339)
            imshow(I[0, ..., 2].copy(), interpolation='none', cmap='bone')
            xticks([]);yticks([])
            subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)


            with tf.gfile.Open(
                'test/test_img-{0}.png'.format(step),
                mode='w+b') as sfile:
                savefig(sfile)
            clf()
        sess.close()
        print(p1/max_steps, p2/max_steps, p3/max_steps)
        print(n1/max_steps, n2/max_steps, n3/max_steps)
if __name__ == '__main__':
    test(1)
