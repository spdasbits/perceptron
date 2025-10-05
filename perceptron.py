# import tensorflow as tf
# class MyDenseLayer(tf.keras.layers.Layer):
#     def __init__(self, input_dim, output_dim):
#         super(MyDenseLayer, self).__init__()
        
#         #initialize weight and bias
#         self.W = self.add_weight([input_dim, output_dim])
#         self.b = self.add_weight([1, output_dim])
#     def call(self, inputs):
#         # forward propagate the inputs
#         z = tf.matmul(inputs, self.W) + self.b
        
#         # Feed through a non - linear activation
#         output = tf.math.sigmoid(z)
#         return output

import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()
        
        # Initialize weight and bias
        self.W = self.add_weight(shape=(input_dim, output_dim),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(output_dim,),
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs):
        # forward propagate the inputs
        z = tf.matmul(inputs, self.W) + self.b
        
        # Feed through a non-linear activation
        output = tf.math.sigmoid(z)
        return output

layer = MyDenseLayer(input_dim=3, output_dim=2)
x = tf.random.normal((5, 3))
# Forward pass
y = layer(x)
print("Output shape:", y.shape)
print("Output:", y.numpy())
