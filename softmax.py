from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 
import time
import data_helpers # contains methods to load the data

# record begin time of execution
begintime = time.time()

# defining the parameters
batch_size = 100
learning_rate = 0.005
max_steps = 1000

# load_data returns the dictionary containing, 
# images_train : the training dataset of 50000 images
# labels_train : lables for the training dataset
# images_test : dataset of 10000 images to test the model
# labels_test : labels of test dataset
# classes : 10 text labels for translating the numbers 0 to 9 into a word like 0 for 'plane', 1 for 'car',etc 
data_sets = data_helpers.load_data()

# Define input placeholders to store the input data and labels
image_placeholders = tf.placeholder(tf.float32, shape=[None,3072])
labels_placeholders = tf.placeholder(tf.int64, shape=[None])

# define variables, wieghts and biases; these are the values that we are supposed to optimize
weights = tf.Variable(tf.zeros([3072,10]))
biases = tf.Variable(tf.zeros([10]))

# logits represents the result of the classifier formed by multiplying input by weights and adding biases
logits = tf.matmul(image_placeholders, weights) + biases

# define loss function (difference between predicted and known label)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels_placeholders))
#global_variables_initializer

# training operation using GradientDescentOptimiser (goal of training is to minimize the loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# operation comparing the prediction of model with true known label
correct_prediction = tf.equal(tf.argmax(logits,1),labels_placeholders)

# calculate the accuracy of the model
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# upto this point entire tensorflow graph is formed 

# Run the tensorflow graph
with tf.Session() as sess:
	# initialize the variables
	sess.run(tf.global_variables_initializer())
	# Repeat Run max_steps times
	for i in range(max_steps):
		# choose random batch_size number of indices
		indices = np.random.choice(data_sets['images_train'].shape[0],batch_size)
		# image_batch and labels_batch contains the images and their labels selected corresponding to the ramdomly selected indices
		image_batch = data_sets['images_train'][indices]
		labels_batch = data_sets['labels_train'][indices]

		# ocasionally print the accuracy of the model
		if i%100 == 0:
			train_accuracy = sess.run(accuracy, feed_dict={image_placeholders: image_batch,labels_placeholders:labels_batch})
			print('Step {:5d}: training accuracy {:g}'.format(i,train_accuracy))

		sess.run(train_step, feed_dict={image_placeholders: image_batch, labels_placeholders:labels_batch})
		# training of model is done

	# test the model for test dataset
	test_accuracy = sess.run(accuracy, feed_dict={image_placeholders: data_sets['images_test'],labels_placeholders: data_sets['labels_test']})
	print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()

print('Total time: {:5.2f}s'.format(endTime - begintime))

