import tensorflow as tf


class FullyConnected:
    """
    Simple fully connected block
    """
    
    def __init__(self, 
        latent_dimension=10,
        hidden_layers=3,
        non_lin='leaky_relu', 
        input_dim=None,
        output_dim=None, 
        name='encoder'):
        
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize list of trainable variables of the layer
        self.trainable_variables = []
        
        # Convert str into an actual tensorflow operator
        if non_lin == 'tanh':
            self.non_lin = tf.tanh
        elif non_lin == 'leaky_relu':
            self.non_lin = tf.nn.leaky_relu

        # Build weights
        self.build()
        
        
        
    def build(self):
        """
        Builds the weights of the layer
        """
        
        # Initialize weights dict
        self.W = {}
        self.b = {} 

        # Iterate over all layers
        for layer in range(self.hidden_layers):

            # Make sure the dimensions are used for the weights
            left_dim = self.latent_dimension
            right_dim = self.latent_dimension
            if (layer == 0) and (self.input_dim is not None):
                left_dim = self.input_dim
            if (layer == self.hidden_layers-1) and (self.output_dim is not None):
                right_dim = self.output_dim

            # Initialize weight matrix
            self.W[str(layer)] = tf.get_variable(name='W_'+self.name+'_{}'.format(layer),
                shape=[left_dim, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
                trainable=True,
                dtype=tf.float32)
            self.trainable_variables.append(self.W[str(layer)])

            # Initialize bias vector
            self.b[str(layer)] = tf.get_variable(name='b_'+self.name+'_{}'.format(layer),
                shape=[1, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
                trainable=True,
                dtype=tf.float32)
            self.trainable_variables.append(self.b[str(layer)])

                            
    def __call__(self, h):

        for layer in range(self.hidden_layers):
            if layer==self.hidden_layers-1:
                h = tf.matmul(h, self.W[str(layer)]) + self.b[str(layer)]
            else:
                h = self.non_lin(tf.matmul(h, self.W[str(layer)])+ self.b[str(layer)])

        return h