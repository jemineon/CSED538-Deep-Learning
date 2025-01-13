class DCMLP(tf.Module):
	def __init__(self, layers):
		self.layers = layers

	@tf.function
	def __call__(self, x):
		for layer in self.layers:
			res = x
			out = layer(x)
			x = tf.concat([out, res], axis=-1)
			# print(f"shape of new input: {x.get_shape()}")
		return out

# Create model
hidden_layer1_size = 16
hidden_layer2_size = 2
output_size = 10

mlp_model = DCMLP([
	DenseLayer(out_dim=hidden_layer1_size, activation=tf.nn.relu),
	DenseLayer(out_dim=hidden_layer2_size, activation=tf.nn.relu),
	DenseLayer(out_dim=output_size)
])