import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

sum=tf.add(a,b)
sess = tf.Session()
print(sess.run(sum,{a:2,b:3}))