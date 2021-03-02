import tensorflow as tf

a=tf.convert_to_tensor(1)

b=tf.multiply(a,3)
sess=tf.Session()

replace_dict={a:15}

print(sess.run(b,feed_dict=None))
print(sess.run(b,feed_dict=replace_dict))

