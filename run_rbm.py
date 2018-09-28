#!/usr/bin/env python

import os, sys, rbm, multlogistic, h5py
import numpy as np

tmpdir = r"./weights"
np.random.seed(1)

def make_dataset(h5_location):
    f = h5py.File(h5_location,'r')
    images = np.array(f['features'])
    imShape = (images.shape[2], images.shape[3])
    images = images.reshape(images.shape[0], images.shape[2]*images.shape[3])/255.
    labels = np.array(f['targets'])
    labels = labels.reshape(labels.shape[0])
    f.close()
    return images, labels, imShape

images_train, labels_train, imShape = make_dataset("data/mnist_train.hdf5")
images_real,  labels_real,  imShape = make_dataset("data/mnist_test.hdf5")


# Define the RBM
rbm = rbm.RBM(images_train.shape[1], 150, actType='Logistic', batchSize=500)

exit(-1)

# Train the RBM
rbm.learn(images_train, maxIter=2, rate=0.01, wDecay=0.01)
#rbm.learn(images_train, maxIter=2000, rate=0.01, wDecay=0.01)

# Save the model
rbm.write(os.path.join(tmpdir, "rbmb.pickle"))

# View the image
rbm.viewWeights(imShape, outfile=os.path.join(tmpdir, r"weights.png"))

# Use hidden layer as input to multinomial regression.
# Here we map each training image to the feature space
hidden_train = rbm.dopass(images_train, forward=True)
hidden_real = rbm.dopass(images_real, forward=True)

print("Training multilogistic regression...")

# Draw from a normal distribution the # of hidden units*num_classes
t0 = 0.005 * np.random.randn(hidden_train.shape[1] * 10)

# Get the number of classes
nc = int(t0.size/hidden_train.shape[1])

# Multi-Logistic
t = multlogistic.logist_reg(hidden_train, labels_train, multlogistic.mult_logist_cost, t0, lamb=1e-4, nc=nc, maxiter=300, disp=False)

print("Predicting results...")

prob = multlogistic.mult_logist_predict(t.reshape(nc, hidden_real.shape[1]), hidden_real, nc)
pred = np.argmax(prob, axis=0)

diff = labels_real - pred

[ print(diff[i]) for i in range(1) ]
wh = (diff == 0)
print("# data: %s, accuracy: %s" % (diff.size, np.mean(wh)))

