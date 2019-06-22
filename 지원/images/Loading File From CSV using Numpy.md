# Loading data from file using Numpy

| Exam1 | Exam2 | Exam3 | Final |
| ----- | ----- | ----- | ----- |
| 73    | 80    | 75    | 152   |
| 93    | 88    | 93    | 185   |
| 89    | 91    | 90    | 180   |
| 96    | 98    | 100   | 196   |
| 73    | 66    | 70    | 142   |
| 53    | 46    | 55    | 101   |

```python
import numpy as np

xy = np.loadtxt('FILENAME.csv', delimeter=',', dtype=np.float32)
X = xy[:, 0:-1]
Y = xy[:, [-1]]

# Make sure the shape and data are OK
print(X.shape, X, len(X))
print(Y.shape, Y, len(Y))

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
  cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={ X: X_data, Y: Y_data })
  if step % 10 == 0:
    print(step, f"Cost: {cost_val}", f"\nPrediction:\n{hy_val}")
```

## Queue Runners

```python
import tensorflow as tf

filename_queue = tf.train.string_input_producer(
	['1.CSV', '2.CSV', ...],
  shuffle=False, name='filename_queue'
)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# Collect batches of csv in
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
sess = tf.Session()

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
  x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
  
coord.request_stop()
coord.join(threads)
```

