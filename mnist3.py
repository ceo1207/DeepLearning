import tensorflow as tf
import input_data
# use conv layer to recognize hand-written numbers


def weightVariable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def biasVariable(shape):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init)


input = tf.placeholder(tf.float32, shape=[None, 784])
truth = tf.placeholder(tf.float32, shape=[None, 10])

# set up the network
# conv1 variable
filter1 = weightVariable([5,5,1,32])
# batchsize height weight channels
inputImage = tf.reshape(input, [-1, 28, 28, 1])
conv1 = tf.nn.conv2d(inputImage, filter1, strides=[1,1,1,1], padding="SAME")
conv1 = tf.nn.relu(conv1+biasVariable([32]))
pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
# conv2 Variable
filter2 = weightVariable([5,5,32,64])
conv2 = tf.nn.conv2d(pool1, filter2, strides=[1,1,1,1], padding="SAME")
conv2 = tf.nn.relu(conv2+biasVariable([64]))
pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
# fully connected
pool2Flat = tf.reshape(pool2, [-1, 7*7*64])
w1 = weightVariable([7*7*64,1024])
b1 = biasVariable([1024])
fc1 = tf.nn.relu(tf.matmul(pool2Flat,w1)+b1)
dropPlace = tf.placeholder(tf.float32)
fc1Drop = tf.nn.dropout(fc1, dropPlace)
w2 = weightVariable([1024,10])
b2 = biasVariable([10])
fc2 = tf.nn.relu(tf.matmul(fc1Drop,w2)+b2)
output = tf.nn.softmax(fc2)

# train
loss = -tf.reduce_sum(truth*tf.log(output))
train = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

# test
result = tf.equal(tf.argmax(truth,1),tf.argmax(output,1))
accuracy = tf.reduce_mean(tf.cast(result,tf.float32))

sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)
mnist = input_data.read_data_sets('data/', one_hot=True)
for i in range(100000):
    batch = mnist.train.next_batch(50)
    sess.run(train, feed_dict={input:batch[0],truth:batch[1],dropPlace:0.5})
    if i%100 == 0 :
        print sess.run(accuracy, feed_dict={input:batch[0],truth:batch[1],dropPlace:1.0})

sess.close()
