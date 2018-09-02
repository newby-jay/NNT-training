from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from pylab import *
import os
import subprocess
import tensorflow as tf
from itertools import cycle
from matplotlib.pylab import histogram
from scipy import interpolate
from itertools import product, cycle, count, permutations, ifilter, repeat, izip, imap
from multiprocessing import Pool
import tempfile
from cStringIO import StringIO
from PIL import Image
import pims
import uuid

Nframes = 5
imageSize = 256
ximg0 = array([[[i, j] for i in arange(imageSize)] for j in arange(imageSize)]) # image pixel coordinates
X0, Y0 = meshgrid(arange(imageSize) - imageSize/2.,
                arange(imageSize) - imageSize/2.)

def rand(*args):
    #shape = tf.to_int32(tf.squeeze(args))
    return tf.random_uniform(args)
def makeVid():

    ximg = tf.reshape(
        tf.constant(float32(ximg0)),
        [1, 1, imageSize, imageSize, 2]
        )
    X = tf.constant(float32(X0))
    Y = tf.constant(float32(Y0))
    def genParticlePaths(Nparticles, Dlist):
        dxi = tf.transpose(
            tf.sqrt(2*Dlist)*tf.random_normal((Nframes, 3, Nparticles)),
            [0, 2, 1])
        xmarg = imageSize/6
        xy0 = -xmarg + (imageSize + 2*xmarg)*rand(Nparticles, 2)
        z0 = tf.random_uniform([Nparticles, 1])*60.
        x0 = tf.concat([xy0, z0], 1)
        return x0 + tf.cumsum(dxi, axis=0)
    def genVideo(r, z, kappa, a, I0, I1, IbackLevel, particleType):
        uw = (0.5 + rand(1))/2.
        un = tf.floor(3*rand(1))
        uampRing = 0.2 + 0.8*rand(1)
        ufade = 15 + 10*rand(1)
        rmax = ufade*(un/uw)**(2./3.)
        ufadeMax = 0.85 #+ 0.05*rand(1)
        fade = uamp*(1. - ufadeMax*tf.abs(tf.tanh(z/ufade)))
        def ring1():
            core = tf.exp(-(r**2/(8.*a))**2)
            ring = uampRing*tf.sin(uw*pi*(r/ufade)**(1.5)*tf.to_float(r<rmax))**2
            return ring, core
        def ring2():
            core = tf.exp(-(r**2/(8.*a))**2)
            ring = tf.exp(-(r - z)**4/(a)**4) + uampRing*0.5*tf.to_float(r<z)
            return ring, core
        def ring3():
            core = tf.exp(-(r**2/(8.*a**2))**2)
            ring = tf.zeros_like(r)
            return ring, core
        ring, core = tf.case(
            [(particleType <= 1, ring1),
            (particleType >= 3, ring3)],
            default=ring2
        )
        I = tf.transpose(
            tf.reduce_sum(
                fade*(core + ring),
                axis=3),
            [2, 0, 1]) # Nt, Ny, Nx
        I += IbackLevel*I1*tf.sin(
            rand(1)*6*pi/512*tf.sqrt(
                rand(1)*(X - rand(1)*512)**2 + rand(1)*(Y - rand(1)*512)**2))
        I += tf.random_normal((Nframes, imageSize, imageSize), stddev=kappa)
        return I

    particleType = tf.floor(rand(1)[0]*5)
    I0 = 0. # min base intensity
    I1 = 1. # max base intensity
    a = 1.5 + rand(1)[0]*4.5 # spot radius scale factor
    kappa = 0.1*rand(1)[0] # noise level
    # a = 1 + mx
    # x = (a - 1)/m
    # a2 = 1 + m2x
    # x = (a2 - 1)/m2
    # a2 = (a - 1)m2/m + 1
    # (a - 1)*5./3.5 + 1.
    def P1():
        Np = tf.to_int32(tf.maximum(1., tf.round(50*rand(1)[0])))
        uamp = (1. + 3.*rand(1)[0]) + 0.1*(rand(Np)*2.-1.)
        return Np, uamp
    def P2():
        Np = tf.to_int32(tf.maximum(1., tf.round(50*rand(1)[0])))
        uamp = (1. + 3.*rand(1)[0]) + 0.1*(rand(Np)*2.-1.)
        return Np, uamp
    def P3():
        ap = a#(a - 1)*7./3.5 + 1.
        Np = tf.to_int32(tf.maximum(1., tf.round(150/ap*rand(1)[0])))
        uamp = 1. + 0.1*(rand(Np)*2.-1.)
        Ncut = tf.to_int32(rand(1)[0]*tf.to_float(Np/4))
        uamp = tf.concat([10*rand(1)*uamp[:Ncut], uamp[Ncut:]], 0)
        return Np, uamp
    Nparticles, uamp = tf.case(
        [(particleType <= 1, P1),
        (particleType >= 3, P3)],
        default=P2
    )

    Dlist = 0.1 + rand(Nparticles)*10. # diffusivities
    IbackLevel = rand(1)**2*0.15
    xi = genParticlePaths(Nparticles, Dlist) # random brownian motion paths
    XALL = tf.transpose(
        tf.tile(ximg, [Nframes, Nparticles, 1, 1, 1]),
        [2, 3, 0, 1, 4]) \
        - xi[..., :2]
    r = tf.sqrt(XALL[..., 0]**2 + XALL[..., 1]**2)
    I = genVideo(r, xi[..., 2], kappa, a, I0, I1, IbackLevel, particleType)
    # simulate 16bit integer data
    Imin = tf.reduce_min(I)
    Imax = tf.reduce_max(I)
    I = (I - Imin)/(Imax - Imin)
    I = tf.round(I*tf.maximum(256., tf.round(2**16*rand(1))))
    # normalize to mean zero, unit std
    Imean, Ivar = tf.nn.moments(I, [0, 1, 2])
    I = (I - Imean)/tf.sqrt(Ivar)
    I = tf.reshape(
        tf.transpose(I, [1, 2, 0]),
        [1, imageSize, imageSize, Nframes])
    ## compute, for each pixel, the distance to the nearest particle (expensive and slow)
    sigma = 4.
    detectors = tf.reduce_sum(
        tf.to_int32(r < sigma),
        axis=3
    )
    P = tf.transpose(
        tf.to_int32(detectors > 0),
        [2, 0, 1])
    labels = tf.stack([1-P, P], 3)
    return I, labels
if __name__ == '__main__':
    Nframes = 3
    with tf.Graph().as_default():
        V, labels = makeVid()
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        for n in arange(32):
            vid = sess.run(V)
            img = vid[0, ..., 1]
            img -= img.min()
            img /= img.max()
            img = uint8(around(255*img))
            with tf.gfile.Open(
                'synthTest/synth-{0}.png'.format(n+1),
                mode='w+b') as sfile:
                Image.fromarray(img).save(sfile, 'png')
