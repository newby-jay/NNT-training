from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import tensorflow as tf
from numpy import *
from scipy import rand

MOVING_AVERAGE_DECAY = 0.999


def nonLinearity(conv, biases):
    """Nonlinear mapping applied to convolution output."""
    return tf.nn.softplus(tf.nn.bias_add(conv, biases))

def makeKernel(shape, name='weights', dprob=1., trainable=True):
    """Initialize a convolution kernal with dropout."""
    kernel = tf.get_variable(
        name,
        shape,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return kernel if dprob==1 else tf.nn.dropout(kernel, dprob)

def makeBiases(nChannels, name='biases', bias_init=0.001, trainable=True):
    """Initialize a vector of biases."""
    return tf.get_variable(
        name,
        [nChannels],
        initializer=tf.constant_initializer(bias_init),
        trainable=trainable)

def network(currentFrame, previousFrame, batch_size=1, dprob=1., tflag=True):
    """Contruct and initialize the neural network."""
    imagesShape = tf.shape(currentFrame)
    imgSize = tf.slice(imagesShape, [1], [2])
    ## conv1
    with tf.variable_scope('conv1') as scope:
        kernel1 = makeKernel(
            shape=[9, 9, 1, 9], name='weights1', dprob=dprob, trainable=tflag)
        biases = makeBiases(9, trainable=tflag)
        conv1 = tf.nn.convolution(
            currentFrame, kernel1, strides=[2, 2], padding='SAME')
        toConv2 = nonLinearity(conv1, biases)
    with tf.variable_scope('conv2') as scope:
        kernel1 = makeKernel(
            shape=[7, 7, 9, 6], name='weights1', dprob=dprob, trainable=tflag)
        kernel2 = makeKernel(
            shape=[7, 7, 6, 6], name='weights2', dprob=dprob, trainable=tflag)
        kernel3 = makeKernel(
            shape=[3, 3, 9, 6], name='weights3', dprob=dprob, trainable=tflag)
        biases = makeBiases(18, trainable=tflag)
        biasesRNN = makeBiases(6, name='biasesRNN', trainable=tflag)
        conv1 = tf.nn.convolution(toConv2, kernel1, padding='SAME')
        preConv = tf.nn.convolution(
            nonLinearity(conv1, biasesRNN),
            # conv1,
            kernel2,
            padding='SAME',
            dilation_rate=[2, 2])
        toNextFrame = tf.nn.convolution(
            preConv, kernel2, padding='SAME', dilation_rate=[3, 3])
        conv3 = tf.nn.convolution(toConv2, kernel3, padding='SAME')
        conv = tf.concat([previousFrame, conv1, conv3], 3)
        toConv3 = nonLinearity(conv, biases)
    with tf.variable_scope('conv3') as scope:
        kernel = makeKernel([5, 5, 18, 2], dprob=dprob, trainable=tflag)
        biases = makeBiases(2, trainable=tflag)
        conv = tf.nn.convolution(toConv3, kernel, padding='SAME')
        toIntrp1 = nonLinearity(conv, biases)
    ################################################################
    ## interpolate 1
    # with tf.variable_scope('intrp1') as scope:
        # Ikernel1 = makeKernelinterp([3, 3, 5, 5], name='Iweights1', dprob=1, trainable=tflag)
        # finalOut = convolutionTranspose(toIntrp1, Ikernel1, [1, 2, 2, 1], img_size, channel_size = 5)
    finalOut = tf.image.resize_images(toIntrp1, imgSize)
    return finalOut, toNextFrame
################################################################
################################################################
################################################################
def processFrames(images, Nframes=5, batch_size=1, dprob=1., tflag=True):
    imagesShape = tf.shape(images)
    imgSize = tf.reshape(tf.slice(imagesShape, [0], [3]), [-1])
    image_shape = tf.concat([imgSize, [1]], 0)
    with tf.variable_scope('RNN') as scope:
        logitsList = []
        toNextFrameList = []
        toNextFrameList.append(tf.zeros((1, 128, 128, 6)))
        for n in arange(Nframes):
            if n > 0:
                tf.get_variable_scope().reuse_variables()
            logits, toNextFrame = network(
                tf.slice(images, [0, 0, 0, n], image_shape),
                toNextFrameList[n],
                dprob=dprob)
            logitsList.append(logits)
            toNextFrameList.append(toNextFrame)
        for n in arange(Nframes)[::-1][1:]:
            tf.get_variable_scope().reuse_variables()
            logits, toNextFrame = network(
                tf.slice(images, [0, 0, 0, n], image_shape),
                toNextFrameList[n+1],
                dprob=dprob)
            logitsList[n] += logits
            toNextFrameList[n] = toNextFrame
        # # frame1 = tf.stack([images[..., 2], images[..., 0], images[..., 1]], 3)
        # frame2 = tf.slice(images, [0, 0, 0, 1], image_shape)
        # # frame2 = images
        # logit2, toNextFrame2 = network(frame2, toNextFrame1, dprob=dprob)
        # tf.get_variable_scope().reuse_variables()
        # frame3 = tf.slice(images, [0, 0, 0, 2], image_shape)
        # # frame3 = tf.stack([images[..., 1], images[..., 2], images[..., 0]], 3)
        # logit3, toNextFrame3 = network(frame3, toNextFrame2, dprob=dprob)
    return tf.concat(logitsList, 0)
################################################################
################################################################
################################################################

def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits, [-1, 2]),
        labels=tf.reshape(labels, [-1, 2]),
    name='cross_entropy_per_example')
    cross_entropy_reduced = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_reduced)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.999, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
def train(total_loss, global_step, learningRate, decayFactor):
    lr = tf.train.exponential_decay(
        learningRate,
        global_step,
        50000,
        decayFactor,
        staircase=True)
    tf.summary.scalar('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    ## Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #   tf.histogram_summary(var.op.name, var)
    ## Add histograms for gradients.
    # for grad, var in grads:
    #   if grad is not None:
    #     tf.histogram_summary(var.op.name + '/gradients', grad)
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,
        global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op, lr
