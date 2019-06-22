# Linear Regression

## Concepts

### Predicting exam score: regression

| x (hours) | y (score) |
| --------- | --------- |
| 10        | 90        |
| 9         | 80        |
| 3         | 50        |
| 2         | 30        |

> 종속 변수 **y** 와 한 개 이상의 독립 변수 **x** 와의 선형 상관 관계를 모델링하는 회귀분석 기법
>
> > H(x) = Wx + b
>
> - 한 개의 설명 변수에 기반한 경우 **단순 선형 회귀**
> - 둘 이상의 설명 변수에 기반한 경우 **다중 선형 회귀**
>
> 선형 회귀는 선형 예측 함수를 사용해 회귀식을 모델링하며, 알려지지 않은 파라미터의 경우 데이터로부터 추정한다. 이렇게 만들어진 회귀식을 **선형 모델** 이라고 정의한다.

#### Cost (Loss) Function

> How fit the line to our training data
>
> H(x) - y

W, b 값이 학습 데이터를 얼마나 잘 표현하는지를 계량하기 위해 Cost Function을 이용.

![Cost Func](./images/graph.png)

![Definition of Cost Func](./images/definition.png)

> #### Goal
>
> > Minimize cost(W,b)



## Materialization with Tensorflow

```python
import tensorflow as tf

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis Wx+b
hypothesis = W * x_train + b

# Cost (Loss) Function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
```

> #### **GradientDescent**
>
> ```python
> # Minimize
> optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
> train = optimizer.minimize(cost)
> ```

```python
# Launch the graph in a session.
sess = tf.Session()
# Initialized global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(cost), sess.run(W), sess.run(b))
```

> #### Using Placeholder
>
> ```python
> X = tf.placeholder(tf.float32, shape=[None])
> Y = tf.placeholder(tf.float32, shape=[None])
> 
> for step in range(2001):
>   cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1,2,3], Y: [1,2,3]})
>   if step % 20 == 0:
>     print(step, cost_val, W_val, b_val)
> ```
>
> 