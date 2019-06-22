# Tensorflow

- An open source software library for numerical computation using data flow graphs.
- Based on `Python`

## What is a Data Flow Graph

- Nodes in the graph represent mathematical operations
- Eges represent the multi-dimensional data arrays (tensors) communicated between them.



## Installation

```bash
sudo -H pip3 install --upgrade tensorflow
```

### Check installation and version

```python
import tensorflow as tf
print(tf.__version__) # 2.0.0-beta1
```



## Basic Usage

```python
import tensorflow as tf

# Create a constant op
# this op is added as a node to the default graph
hello = tf.constant('Hello, Tensorflow!')

# Seart a TF session
sess = tf.Session()

# Run the op and get result
print(sess.run(hello))
```



### Computational Graph

```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print('node1:', node1, 'node2:', node2)
print('node3:', node3)

sess = tf.Session()
print('sess.run(node1, node2): ', sess.run([node1, node2])) # [3.0, 4.0]
print('sess.run(node3): ', sess.run(node3)) # 7.0
```



> ### TensorFlow Mechanics
>
> 1. Build graph using TensorFlow operations
> 2. Feed data and run graph (operation) **sess.run(op, feed_dict={ x: x_data })**
> 3. Update variables in the graph and return values



### Placeholder

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # provide a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={ a: 3, b: 4.5 })) # 7.5
print(sess.run(adder_node, feed_dict={ a: [1,3], b: [2,4] })) # [ 3. 7.]
```



### Everything is **Tensor**

```python
3 # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 4.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

#### Tensor Ranks, Shapes, and Types

```python
t = [[1,2,3], [4,5,6], [7,8,9]]
```

| Rank | Shape             | Math Entity | Python Example                         |
| ---- | ----------------- | ----------- | -------------------------------------- |
| 0    | []                | 0-D         | A 0-D tensor. A scalar.                |
| 1    | [D0]              | 1-D         | A 1-D tensor with shape [5].           |
| 2    | [D0, D1]          | 2-D         | A 2-D tensor with shape [3,4].         |
| 3    | [D0, D1, D2]      | 3-D         | A 3-D tensor with shape [1,4,3].       |
| n    | [D0, D1, …, Dn-1] | n-D         | A tensor with shape [D0, D1, …, Dn-1]. |

| Data type | Python Type | Description             |
| --------- | ----------- | ----------------------- |
| DT_FLOAT  | tf.float32  | 32 bits floating point. |
| DT_DOUBLE | tf.float64  | 64 bits floating point. |
| DT_INT8   | tf.int8     | 8 bits signed integer.  |
| DT_INT16  | tf.int16    | 16 bits signed integer. |
| DT_INT32  | tf.int32    | 32 bits signed integer. |
| DT_INT64  | tf.int64    | 64 bits signed integer. |

