import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

sess = tf.Session()

# init = tf.global_variables_initializer()
# sess.run(init)
init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)



# print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

print(sess.run(loss, {x:[4,2], y:[3,2]}))

# print(sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# sess.run(init) # reset values to incorrect defaults.
# for i in range(1000):
#   sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
#
# print(sess.run([W, b]))