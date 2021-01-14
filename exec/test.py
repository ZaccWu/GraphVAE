import tensorflow as tf
a = tf.placeholder(tf.float32, shape=None, name='a')
b = tf.placeholder(tf.float32, shape=None, name='b')

output = tf.add(a,b)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={a:[7,1],b:[2,3]}))