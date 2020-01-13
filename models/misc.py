import tensorflow as tf


@tf.custom_gradient
def scatter_nd_add_diff(x0, x1, x2):
    """
    Custom gradient version of scatter_nd_add
    """
    dummy = tf.Variable(x0, name='dummy', use_resource=True)
    reset_dummy = dummy.assign(0.0*x0)

    with tf.control_dependencies([reset_dummy]):
        f = tf.scatter_nd_add(dummy, x1, x2)

    def grad(dy, variables=[dummy]):
        g = tf.gather_nd(dy, x1)
        return [None, None, g], [None]

    return f, grad

@tf.custom_gradient
def v_check_control(V_update, V_consigne, gen_indices):
    
    gen_mask = tf.sparse_to_dense(gen_indices,
                                  tf.shape(V_update),
                                  0.0,
                                  default_value=1.0,
                                  validate_indices=False)
    value_mask =  tf.sparse_to_dense(gen_indices, 
                                     tf.shape(V_update),
                                     tf.reshape(V_consigne, [-1]),
                                     default_value=0.0,
                                     validate_indices=False)

    f = gen_mask * V_update + value_mask

    def grad(dy):
        
        g1 = tf.sparse_to_dense(gen_indices,
                              tf.shape(V_update),
                              0.0,
                              default_value=1.0,
                              validate_indices=False)
        g1 = g1 * dy
        g2 = tf.gather_nd(dy, gen_indices)
        g2 = tf.reshape(g2, tf.shape(V_consigne))

        return [g1, g2, None]

    return f, grad

def build_indices(connection, dim, num_samples):

    indices = tf.expand_dims(tf.range(0, num_samples, 1),-1)
    indices = tf.reshape(tf.tile(indices,[1,dim]), [-1])
    indices = tf.stack([indices, tf.reshape(connection, [-1])], 1)

    return indices
