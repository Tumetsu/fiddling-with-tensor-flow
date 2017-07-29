import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create a symbolic variable for creating a graph to describe the operations we need to do outside
# Python for efficient computing
x = tf.placeholder(tf.float32, [None, 784])

# Model parameters are passed as variables. In this case Weights W and biases b.
# W and b are going to be learned so they can initially be 0.

# Notice that W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it to produce
# 10-dimensional vectors of evidence for the difference classes. b has a shape of [10] so we can add it to the output.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# First, we multiply x by W with the expression tf.matmul(x, W). This is flipped from when we multiplied them in our
# equation, where we had Wx, as a small trick to deal with x being a 2D tensor with multiple inputs. We then add b,
# and finally apply tf.nn.softmax. The model is done.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# A new placeholder is needed for input correct answers. Required for cross-entropy check.
y_ = tf.placeholder(tf.float32, [None, 10])

# First, tf.log computes the logarithm of each element of y. Next, we multiply each element of y_ with the
# corresponding element of tf.log(y). Then tf.reduce_sum adds the elements in the second dimension of y, due to the
# reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning
# rate of 0.5. Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit
# in the direction that reduces the cost
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Each step of the loop, we get a "batch" of one hundred random data points from our training set. We run train_
# step feeding in the batches data to replace the placeholders.
# Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent.
# Ideally, we'd like to use all our data for every step of training because that would give us a better sense of what
# we should be doing, but that's expensive. So, instead, we use a different subset every time. Doing this is cheap
# and has much of the same benefit.
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# let's figure out where we predicted the correct label. tf.argmax is an extremely useful function which gives you
# the index of the highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the label our model
# thinks is most likely for each input, while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if
# our prediction matches the truth.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Calculate float value of distribution of True and Falses
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Print accuracy. Should be around 90-92% which is bad. Over 97% start to become good. Reason for current accuracy is our
# simple model
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
