import os
import json
import logging
import copy

import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm

from models.misc import scatter_nd_add_diff, v_check_control
from models.layers import FullyConnected
#from module.load import parse_file


# TODO : If required, add support of other standard power grids

class GraphNeuralSolver(object):
    """
    """

    def __init__(self,
        sess,
        latent_dimension=10,
        hidden_layers=3,
        correction_updates=30,
        non_lin='leaky_relu',
        name='graph_neural_solver',
        directory='./',
        model_to_restore=None):

        # Get session and params
        self.sess = sess
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.correction_updates = correction_updates
        self.non_lin = non_lin
        self.name = name
        self.directory = directory
        self.current_train_iter = 0

        self.trainable_variables = []

        # Neural network output normalization. This factor helps the learning process by starting with small
        # reasonable updates at the start of the learning process
        self.scaling_factor = 1e-4

        # Reload config if there is a model to restore
        if (model_to_restore is not None) and os.path.exists(model_to_restore):
            
            logging.info('    Restoring model from '+model_to_restore)
            path_to_config = os.path.join(model_to_restore, 'config.json')
            with open(path_to_config, 'r') as f:
                config = json.load(f)
            self.set_config(config)

        # Build weights of neural network blocks
        self.build_weights()

        # Build dicts that store all the variables of the architecture
        self.build_dicts()

        # NOTE : ça doit créer des variables qui seront modifiées par un assign
        self.import_power_grid()

        # Build computational graph
        self.build_graph()

        # Restore trained weights if there is a model to restore
        if (model_to_restore is not None) and os.path.exists(model_to_restore):
            saver = tf.train.Saver([*self.trainable_variables, *self.optimizer.variables()])
            path_to_weights = os.path.join(model_to_restore, 'model.ckpt')
            saver.restore(self.sess, path_to_weights)

        # Else randomly initialize weights
        else:
            self.sess.run(tf.initialize_variables(self.trainable_variables))

        # Log config infos
        self.log_config()

    def log_config(self):
        """
        Logs the config of the whole model
        """

        logging.info('    Configuration :')
        logging.info('        Latent dimensions : {}'.format(self.latent_dimension))
        logging.info('        Number of hidden layers per block : {}'.format(self.hidden_layers))
        logging.info('        Number of correction updates : {}'.format(self.correction_updates))
        logging.info('        Non linearity : {}'.format(self.non_lin))
        logging.info('        Current training iteration : {}'.format(self.current_train_iter))
        logging.info('        Model name : ' + self.name)

    def get_config(self):
        """
        Gets the config dict
        """

        config = {
            'latent_dimension': self.latent_dimension,
            'hidden_layers': self.hidden_layers,
            'correction_updates': self.correction_updates,
            'non_lin': self.non_lin,
            'name': self.name,
            'directory': self.directory,
            'current_train_iter': self.current_train_iter
        } 
        return config

    def set_config(self, config):
        """
        Sets the config according to an inputed dict
        """

        self.latent_dimension = config['latent_dimension']
        self.hidden_layers = config['hidden_layers']
        self.correction_updates = config['correction_updates']
        self.non_lin = config['non_lin']
        self.name = config['name']
        self.directory = config['directory']
        self.current_train_iter = config['current_train_iter']

    def save(self):
        """
        Saves the configuration of the model and the trained weights
        """

        # Save config
        config = self.get_config()
        path_to_config = os.path.join(self.directory, 'config.json')
        with open(path_to_config, 'w') as f:
            json.dump(config, f)

        # Save weights
        saver = tf.train.Saver([*self.trainable_variables, *self.optimizer.variables()])
        path_to_weights = os.path.join(self.directory, 'model.ckpt')
        saver.save(self.sess, path_to_weights)

    def build_weights(self):

        self.correction_m = {}
        self.correction_v = {}
        self.correction_theta = {}

        for update in range(self.correction_updates):
            self.correction_m[str(update)] = FullyConnected(
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_correction_m_{}'.format(update),
                input_dim=4+2*self.latent_dimension,
                output_dim=self.latent_dimension
            )
            self.trainable_variables.extend(self.correction_m[str(update)].trainable_variables)

            self.correction_v[str(update)] = FullyConnected(
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_correction_v_{}'.format(update),
                input_dim=4+2*self.latent_dimension,
                output_dim=1
            )
            self.trainable_variables.extend(self.correction_v[str(update)].trainable_variables)

            self.correction_theta[str(update)] = FullyConnected(
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_correction_theta_{}'.format(update),
                input_dim=4+2*self.latent_dimension,
                output_dim=1
            )
            self.trainable_variables.extend(self.correction_theta[str(update)].trainable_variables)

        self.phi = FullyConnected(
            latent_dimension=self.latent_dimension,
            hidden_layers=self.hidden_layers,
            name=self.name+'_phi',
            input_dim=5+2*self.latent_dimension,
            output_dim=self.latent_dimension
        )
        self.trainable_variables.extend(self.phi.trainable_variables)

    def build_graph(self):

        self.discount = tf.Variable(0., trainable=False)
        self.sess.run(tf.initialize_variables([self.discount]))

        

        #### Build variables that will usefull later ####

        # Get line characteristics in polar coordinates
        self.y_ij = 1. / tf.sqrt(self.lines['r']**2 + self.lines['x']**2)
        self.delta_ij = tf.math.atan2(self.lines['r'], self.lines['x'])

        # Build indices
        self.linspace = tf.expand_dims(tf.range(0, self.n_samples, 1),-1)
        self.one_tensor = tf.ones([1], tf.int32)
        self.n_lines_tensor = tf.reshape(self.n_lines,[1])
        self.n_gens_tensor = tf.reshape(self.n_gens,[1])
        self.shape_lines_indices = tf.concat([self.one_tensor, self.n_lines_tensor], axis=0)
        self.shape_gens_indices = tf.concat([self.one_tensor, self.n_gens_tensor], axis=0)

        self.indices_from = tf.reshape(tf.tile(self.linspace, self.shape_lines_indices), [-1])
        self.indices_from = tf.stack([self.indices_from, tf.reshape(self.lines['from'], [-1])], 1)

        self.indices_to = tf.reshape(tf.tile(self.linspace, self.shape_lines_indices), [-1])
        self.indices_to = tf.stack([self.indices_to, tf.reshape(self.lines['to'], [-1])], 1)

        self.indices_gens = tf.reshape(tf.tile(self.linspace, self.shape_gens_indices), [-1])
        self.indices_gens = tf.stack([self.indices_gens, tf.reshape(self.gens['bus'], [-1])], 1)

        # Initialize dummy variables that will be useful for the scatter_nd_add_diff function
        self.n_samples_tensor = tf.reshape(self.n_samples,[1])
        self.n_nodes_tensor = tf.reshape(self.n_nodes,[1])
        self.latent_dim_tensor = tf.reshape(self.latent_dimension,[1])
        self.shape_latent_message = tf.concat([self.n_samples_tensor,
                                               self.n_nodes_tensor,
                                               self.latent_dim_tensor], axis=0)
        self.zero_input = tf.reshape(0.,[1,1,1])
        self.dummy_message = tf.tile(self.zero_input, self.shape_latent_message)

        self.dummy = tf.zeros_like(self.buses['baseKV'])

        #################################################

        #### INITIALIZATION ####
        # Initialize v, theta and latent messages
        self.v = {'0': 1.+tf.zeros_like(self.buses['baseKV'])}
        self.theta = {'0': tf.zeros_like(self.buses['baseKV'])}
        self.latent_message = {'0' : tf.zeros_like(self.dummy_message)}

        # Control the voltage for generators
        self.v['0'] = v_check_control(self.v['0'], self.gens['Vg'], self.indices_gens)
        ########################



        

        # First : perform a compensation by neglecting the losses
        self.sum_p_gen_target_per_sample = tf.reduce_sum(self.gens['Pg']/ self.gens['mbase'] , axis=1, keepdims=True)
        self.sum_p_gen_max_per_sample = tf.reduce_sum(self.gens['Pmax']/ self.gens['mbase'], axis=1, keepdims=True)
        self.sum_p_gen_min_per_sample = tf.reduce_sum(self.gens['Pmin']/ self.gens['mbase'], axis=1, keepdims=True)

        # Ensure that the tension at nodes that have a production are at the right level
        

        # Iterate the neural network local updates
        for update in range(self.correction_updates+1):

            # Send nodes V to lines origins and extremities
            self.v_from[str(update)] = tf.gather(self.v[str(update)], self.lines['from'], batch_dims=1)
            self.v_to[str(update)] = tf.gather(self.v[str(update)], self.lines['to'], batch_dims=1)

            # Send nodes theta to lines origins and extremities
            self.theta_from[str(update)] = tf.gather(self.theta[str(update)], self.lines['from'], batch_dims=1)
            self.theta_to[str(update)] = tf.gather(self.theta[str(update)], self.lines['to'], batch_dims=1)

            # At each line, get the active power leaving the node at the "from" side
            self.p_from_to[str(update)] = self.v_from[str(update)]*self.v_to[str(update)]*self.y_ij/self.lines['ratio']*\
                tf.math.sin(self.theta_from[str(update)] - self.theta_to[str(update)] - self.delta_ij - self.lines['angle']) +\
                self.v_from[str(update)]**2 / self.lines['ratio']**2 * self.y_ij * tf.math.sin(self.delta_ij)

            # At each line, get the active power leaving the node at the "to" side
            self.p_to_from[str(update)] = self.v_from[str(update)]*self.v_to[str(update)]*self.y_ij/self.lines['ratio']*\
                tf.math.sin(self.theta_to[str(update)] - self.theta_from[str(update)] - self.delta_ij + self.lines['angle']) +\
                self.v_to[str(update)]**2 * self.y_ij * tf.math.sin(self.delta_ij)
            
            # At each line, get the reactive power leaving the node at the "from" side
            self.q_from_to[str(update)] = - self.v_from[str(update)]*self.v_to[str(update)]*self.y_ij/self.lines['ratio']*\
                tf.math.cos(self.theta_from[str(update)] - self.theta_to[str(update)] - self.delta_ij - self.lines['angle']) +\
                self.v_from[str(update)]**2 / self.lines['ratio']**2 * (self.y_ij * tf.math.cos(self.delta_ij) - self.lines['b']/2)

            # At each line, get the reactive power leaving the node at the "to" side
            self.q_to_from[str(update)] = - self.v_from[str(update)]*self.v_to[str(update)]*self.y_ij/self.lines['ratio']*\
                tf.math.cos(self.theta_to[str(update)] - self.theta_from[str(update)] - self.delta_ij + self.lines['angle']) +\
                self.v_to[str(update)]**2 * (self.y_ij * tf.math.cos(self.delta_ij) - self.lines['b']/2)
            
            # Compute the active imbalance at each node
            self.delta_p[str(update)] = - scatter_nd_add_diff(self.dummy, self.indices_from, 
                                                             tf.reshape(self.p_from_to[str(update)], [-1])) \
                                        - scatter_nd_add_diff(self.dummy, self.indices_to, 
                                                             tf.reshape(self.p_to_from[str(update)], [-1])) \
                                        - self.buses['Pd'] / self.baseMVA \
                                        - self.buses['Gs'] * self.v[str(update)]**2 / self.baseMVA

            # Compute the reactive imbalance at each node
            self.delta_q[str(update)] = - scatter_nd_add_diff(self.dummy, self.indices_from, 
                                                             tf.reshape(self.q_from_to[str(update)], [-1])) \
                                        - scatter_nd_add_diff(self.dummy, self.indices_to, 
                                                             tf.reshape(self.q_to_from[str(update)], [-1])) \
                                        - self.buses['Qd'] / self.baseMVA \
                                        + self.buses['Bs'] * self.v[str(update)]**2 / self.baseMVA

            


            ### GLOBAL ACTIVE COMPENSATION ###
            self.p_joule[str(update)] = tf.abs(tf.abs(self.p_from_to[str(update)])-tf.abs(self.p_to_from[str(update)])) 

            self.apparent_consumption_per_node[str(update)] = self.buses['Pd'] / self.baseMVA \
                                                              + self.buses['Gs'] * self.v[str(update)]**2 / self.baseMVA

            self.apparent_consumption_per_sample[str(update)] = tf.reduce_sum(self.apparent_consumption_per_node[str(update)], axis=1, keepdims=True)
            self.apparent_consumption_per_sample[str(update)] += tf.reduce_sum(self.p_joule[str(update)], axis=1, keepdims=True)


            self.is_above[str(update)] = tf.math.sigmoid((self.apparent_consumption_per_sample[str(update)] - self.sum_p_gen_target_per_sample)/ \
                (1e-6 * (self.sum_p_gen_max_per_sample - self.sum_p_gen_min_per_sample)))
            self.is_below[str(update)] = 1. - self.is_above[str(update)]

            self.p_gen[str(update)] = self.is_below[str(update)] * ( (self.gens['Pg'] - self.gens['Pmin'])/self.gens['mbase'] \
                    * (self.apparent_consumption_per_sample[str(update)] - self.sum_p_gen_min_per_sample) \
                    / (self.sum_p_gen_target_per_sample - self.sum_p_gen_min_per_sample) \
                    +self.gens['Pmin']/self.gens['mbase']) \
                    + self.is_above[str(update)] * ((self.gens['Pmax'] - self.gens['Pg'])/self.gens['mbase'] \
                    * (self.apparent_consumption_per_sample[str(update)] + self.sum_p_gen_max_per_sample - 2*self.sum_p_gen_target_per_sample) \
                    / (self.sum_p_gen_max_per_sample - self.sum_p_gen_target_per_sample) \
                    +(-self.gens['Pmax'] +2*self.gens['Pg'])/self.gens['mbase'])

            

            self.delta_p[str(update)] += scatter_nd_add_diff(self.dummy, self.indices_gens, 
                                                              tf.reshape(self.p_gen[str(update)], [-1]))

            # Set the reactive generation to locally compensate
            self.q_gen[str(update)] = tf.gather(-self.delta_q[str(update)], self.gens['bus'], batch_dims=1)

            self.delta_q[str(update)] += scatter_nd_add_diff(self.dummy, self.indices_gens, tf.reshape(self.q_gen[str(update)], [-1]))

            #################################

            if update < self.correction_updates : 

                ### NEURAL NETWORK UPDATE ###

                # Building message sum
                self.indices_from_messages = tf.ones([1,self.latent_dimension, 1], dtype=tf.int32) * tf.expand_dims(self.lines['from'], 1)
                self.indices_to_messages = tf.ones([1,self.latent_dimension, 1], dtype=tf.int32) * tf.expand_dims(self.lines['to'], 1)

                self.message_from[str(update)] = tf.gather(tf.transpose(self.latent_message[str(update)], [0,2,1]), 
                                                        self.indices_from_messages, 
                                                        batch_dims=2)
                self.message_from[str(update)] = tf.transpose(self.message_from[str(update)], [0,2,1])

                self.message_to[str(update)] = tf.gather(tf.transpose(self.latent_message[str(update)], [0,2,1]), 
                                                        self.indices_to_messages, 
                                                        batch_dims=2)
                self.message_to[str(update)] = tf.transpose(self.message_to[str(update)], [0,2,1])
                phi_input = tf.stack(
                    [tf.reshape(self.y_ij, [-1]),
                    tf.reshape(self.delta_ij, [-1]),
                    tf.reshape(self.lines['b'], [-1]),
                    tf.reshape(self.lines['ratio'], [-1]),
                    tf.reshape(self.lines['angle'], [-1])],
                    axis=1
                )
                phi_input_from = tf.concat(
                    [phi_input,
                    tf.reshape(self.message_from[str(update)], [-1, self.latent_dimension]),
                    tf.reshape(self.message_to[str(update)], [-1, self.latent_dimension])],
                    axis=1
                )
                phi_input_to = tf.concat(
                    [phi_input,
                    tf.reshape(self.message_to[str(update)], [-1, self.latent_dimension]),
                    tf.reshape(self.message_from[str(update)], [-1, self.latent_dimension])],
                    axis=1
                )

                self.phi_message_from[str(update)] = self.phi(phi_input_from * self.scaling_factor)
                self.phi_message_from[str(update)] = tf.reshape(
                    self.phi_message_from[str(update)], 
                    [self.n_samples, -1, self.latent_dimension]
                )

                self.phi_message_to[str(update)] = self.phi(phi_input_to * self.scaling_factor)
                self.phi_message_to[str(update)] = tf.reshape(
                    self.phi_message_to[str(update)], 
                    [self.n_samples, -1, self.latent_dimension]
                )

                self.message_neighbors[str(update)] = scatter_nd_add_diff(self.dummy_message, self.indices_from, 
                                                                  tf.reshape(self.phi_message_to[str(update)], [-1, self.latent_dimension])) \
                                                    + scatter_nd_add_diff(self.dummy_message, self.indices_to, 
                                                                  tf.reshape(self.phi_message_from[str(update)], [-1, self.latent_dimension]))

                # Correction computation
                correction_input = tf.stack(
                    [tf.reshape(self.v[str(update)], [-1]),
                    tf.reshape(self.theta[str(update)], [-1]),
                    tf.reshape(self.delta_p[str(update)], [-1]),
                    tf.reshape(self.delta_q[str(update)], [-1])],
                    axis=1
                )
                correction_input = tf.concat(
                    [correction_input, 
                    tf.reshape(self.latent_message[str(update)], [-1, self.latent_dimension]), 
                    tf.reshape(self.message_neighbors[str(update)], [-1, self.latent_dimension])],
                    axis=1
                ) 

                delta_v = self.correction_v[str(update)](correction_input* self.scaling_factor)
                delta_v = tf.reshape(delta_v, [self.n_samples, -1]) * self.scaling_factor
                self.v[str(update+1)] = self.v[str(update)] + delta_v

                delta_theta = self.correction_theta[str(update)](correction_input* self.scaling_factor)
                delta_theta =  tf.reshape(delta_theta, [self.n_samples, -1]) * self.scaling_factor
                self.theta[str(update+1)] = self.theta[str(update)] + delta_theta

                delta_message = self.correction_m[str(update)](correction_input* self.scaling_factor)
                delta_message = tf.reshape(delta_message, [self.n_samples, -1, self.latent_dimension])
                self.latent_message[str(update+1)] = self.latent_message[str(update)] + delta_message

                # Control V for the generators, so that it stays at the desired value
                self.v[str(update+1)] = v_check_control(self.v[str(update+1)], self.gens['Vg'], self.indices_gens)            

            # Compute loss
            self.loss[str(update+1)] = tf.reduce_mean(self.delta_p[str(update)]**2 + self.delta_q[str(update)]**2)
            tf.summary.scalar("loss_{}".format(update+1), self.loss[str(update+1)])

            if self.total_loss is None:
                self.total_loss =  self.loss[str(update+1)] * self.discount**((self.correction_updates-update)/5.)
            else:
                self.total_loss +=  self.loss[str(update+1)] * self.discount**((self.correction_updates-update)/5.)

        # Get the final predictions for v, theta, and the loss
        self.v_final = self.v[str(self.correction_updates)]
        self.theta_final = self.theta[str(self.correction_updates)]
        self.loss_final = self.loss[str(self.correction_updates)]

        self.learning_rate = tf.Variable(0., trainable=False)
        self.sess.run(tf.initialize_variables([self.learning_rate]))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gradients, variables = zip(*self.optimizer.compute_gradients(self.total_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 10)
        self.opt_op = self.optimizer.apply_gradients(zip(gradients, variables))
        self.sess.run(tf.initialize_variables(self.optimizer.variables()))


        # Build summary to visualize the final loss in Tensorboard
        tf.summary.scalar("loss_final", self.loss_final)
        self.merged_summary_op = tf.summary.merge_all()

        # Initialize all the dummy variables
        self.sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dummy")))


    def build_dicts(self):
        """
        Defines the dictionnaries that contain all the variables that will be updates throughout the model
        """

        self.delta_p = {}       # local active power mismatch
        self.delta_q = {}       # local reactive power mismatch

        self.v_from = {}        # v gathered at the "from" side of each line
        self.v_to = {}          # v gathered at the "to" side of each line

        self.theta_from = {}    # theta gathered at the "from" side of each line
        self.theta_to = {}      # theta gathered at the "to" side of each line
        
        self.p_from_to = {}     # active power going from the "from" side into the line
        self.p_to_from = {}     # active power going from the "to" side into the line

        self.q_from_to = {}     # reactive power going from the "from" side into the line
        self.q_to_from = {}     # reactive power going from the "to" side into the line

        self.message_from = {}    # messages carried by nodes on the "from" side of every line
        self.message_to = {}    # messages carried by nodes on the "to" side of every line
        self.message_neighbors = {}     # sum of neighboring messages for each node
        self.phi_message_from = {}
        self.phi_message_to = {}
        self.sum_message = {}
        self.global_message = {}

        self.p_joule = {}       # active power lost by Joule's effect
        self.p_gen = {}         # active power produced by each generator
        self.q_gen = {}         # reactive power produced by each generator 

        self.apparent_consumption_per_node = {}     # Consumption at each node, taking all effect into account
        self.apparent_consumption_per_sample = {}   # Total consumption + loss on each sample

        self.is_above = {}      # factor that is 1 if the load is above the generation setpoint
        self.is_below = {}      # factor that is 0 if the load is above the generation setpoint

        self.loss = {}          # Kirchhoff's law violation at each node
        self.total_loss = None  # total loss 

        
    def import_power_grid(self):
        """
        Import files from pypower
        """
        from pypower.case14 import case14
        self.input_data = case14()

        bus_i = np.reshape(self.input_data['bus'][:,0], [1, -1])
        fbus = np.reshape(self.input_data['branch'][:,0], [-1, 1])
        tbus = np.reshape(self.input_data['branch'][:,1], [-1, 1])
        gbus = np.reshape(self.input_data['gen'][:,0], [-1, 1])

        fbus = np.where(bus_i-fbus==0)[1]
        tbus = np.where(bus_i-tbus==0)[1]
        gbus = np.where(bus_i-gbus==0)[1]

        ratios = self.input_data['branch'][:,8] +1.*(self.input_data['branch'][:,8]==0.)

        self.n_nodes = tf.Variable(2*self.input_data['bus'].shape[0], trainable=False, dtype=tf.int32)
        self.n_gens = tf.Variable(self.input_data['gen'].shape[0], trainable=False, dtype=tf.int32)
        self.n_lines = tf.Variable(self.input_data['branch'].shape[0], trainable=False, dtype=tf.int32)

        self.baseMVA = tf.Variable(self.input_data['baseMVA']*np.ones([1,1]), trainable=False, dtype=tf.float32)

        self.buses_default = {}

        zero = np.zeros([1, 2*self.input_data['bus'].shape[0]])
        self.buses_default['baseKV'] = tf.Variable(zero, trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Pd'] = tf.Variable(zero, trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Qd'] = tf.Variable(zero, trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Gs'] = tf.Variable(zero, trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Bs'] = tf.Variable(zero, trainable=False, dtype=tf.float32)#, validate_shape=False)


        self.lines_default = {}
        self.lines_default['from'] = tf.Variable(np.reshape(fbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.lines_default['to'] = tf.Variable(np.reshape(tbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.lines_default['r'] = tf.Variable(np.reshape(self.input_data['branch'][:,2], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['x'] = tf.Variable(np.reshape(self.input_data['branch'][:,3], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['b'] = tf.Variable(np.reshape(self.input_data['branch'][:,4], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['ratio'] = tf.Variable(np.reshape(ratios, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['angle'] = tf.Variable(np.reshape(self.input_data['branch'][:,9], [1, -1])*np.pi/180, trainable=False, dtype=tf.float32)#, validate_shape=False)

        self.gens_default = {}
        self.gens_default['bus'] = tf.Variable(np.reshape(gbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.gens_default['Vg'] = tf.Variable(np.reshape(self.input_data['gen'][:,5], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pg'] = tf.Variable(np.reshape(self.input_data['gen'][:,1], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pmin'] = tf.Variable(np.reshape(self.input_data['gen'][:,9], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pmax'] = tf.Variable(np.reshape(self.input_data['gen'][:,8], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['mbase'] = tf.Variable(np.reshape(self.input_data['gen'][:,6], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)


        self.sess.run(tf.initializers.variables(self.buses_default.values()))
        self.sess.run(tf.initializers.variables(self.lines_default.values()))
        self.sess.run(tf.initializers.variables(self.gens_default.values()))
        self.sess.run(tf.initializers.variables([self.n_nodes,
                                                 self.n_gens,
                                                 self.n_lines,
                                                 self.baseMVA]))


        self.n_samples = tf.Variable(1, trainable=False)

        self.one_dim = tf.Variable(1, trainable=False)
        self.sess.run(tf.initialize_variables([self.n_samples, self.one_dim]))
        self.duplicator = tf.ones(tf.stack([self.n_samples, self.one_dim]), dtype=tf.float32)

        self.buses = {}
        for key, value in self.buses_default.items():
            self.buses[key] = self.buses_default[key] * self.duplicator

        self.lines = {}
        for key, value in self.lines_default.items():
            if key in ['from', 'to']:
                self.lines[key] = self.lines_default[key] * tf.cast(self.duplicator, tf.int32)
            else:
                self.lines[key] = self.lines_default[key] * self.duplicator

        self.gens = {}
        for key, value in self.gens_default.items():
            if key in ['bus']:
                self.gens[key] = self.gens_default[key] * tf.cast(self.duplicator, tf.int32)
            else:
                self.gens[key] = self.gens_default[key] * self.duplicator


    def train(self, 
        max_iter=10, 
        minibatch_size=10, 
        learning_rate=3e-4, 
        discount=0.9,
        p_topo=0.,
        sigma_inj=0.,
        data_directory='data/',
        save_step=None):
        """
        Performs a training process while keeping track of the validation score
        """

        # Log infos about training process
        logging.info('    Starting a training process :')
        logging.info('        Max iteration : {}'.format(max_iter))
        logging.info('        Minibatch size : {}'.format(minibatch_size))
        logging.info('        Learning rate : {}'.format(learning_rate))
        logging.info('        Discount : {}'.format(discount))
        logging.info('        Proba of topo change : {}'.format(p_topo))
        logging.info('        Spread of randomness for inputs : {}'.format(sigma_inj))
        logging.info('        Training data : {}'.format(data_directory))
        logging.info('        Saving model every {} iterations'.format(save_step))

        # Build writer dedicated to training for Tensorboard
        self.training_writer = tf.summary.FileWriter(
            os.path.join(self.directory, 'train'), 
            graph=tf.get_default_graph()
        )

        # Load dataset
        file_list = ['_N_loads_p', '_N_loads_q', '_N_prods_p', '_N_prods_v']

        data = {}
        for file in file_list:
            data[file] = None

        for chronics_dir in os.listdir(data_directory):
            try:
                for file in file_list:
                    path = os.path.join(os.path.join(data_directory, chronics_dir), file+'.csv')
                    if data[file] is None:
                        data[file] = pd.read_csv(path, sep=';')
                    else:
                        print("appending")
                        print(data[file].values.shape)
                        print(pd.read_csv(path, sep=';'))
                        data[file] = pd.concat([data[file], pd.read_csv(path, sep=';')])
                        print(data[file].values.shape)
            except:
                pass
        print(data)

        #for file in file_list:
        #    data[file] = None
        #    for data_id in range(50):
        #        data_directory = '0000{}'.format(data_id)
        #        data_directory = os.path.join('/Users/balthazardonon/Documents/PhD/Code/L2RPN/public_data/datasets/chronics', data_directory[-4:])
        #        path = os.path.join(data_directory, file+'.csv')
        #        if data[file] is None:
        #            data[file] = pd.read_csv(path, sep=';')
        #        else:
        #            data[file].append(pd.read_csv(path, sep=';'))


        self.sess.run(self.n_samples.assign(minibatch_size))



        


        name_to_index_load = {
            '2_C-10.61': 1,
            '3_C151.15': 2,
            '4_C-9.47': 3,
            '5_C201.84': 4,
            '6_C-6.27': 5,
            '9_C130.49': 8,
            '10_C228.66': 9,
            '11_C-138.89': 10,
            '12_C-27.88': 11,
            '13_C-13.33': 12,
            '14_C63.6': 13 
        }

        name_to_index_prod = {
            '1_G137.1': 0,
            '2_G-56.47': 1,
            '3_G36.31': 2,
            '6_G63.29': 3,
            '8_G40.43': 4,
        }

        vn_kv_prod = {
            '1_G137.1': 100.,#135.0, 
            '2_G-56.47': 100.,#135.0, 
            '3_G36.31': 100.,#135.0, 
            '6_G63.29': 100.,#0.208,
            '8_G40.43': 100.,#12.0
        }

        default_gens_bus = {
            0: 0, 
            1: 1, 
            2: 2, 
            3: 5, 
            4: 7
        }

        default_lines_from_bus = {
            0: 0,  
            1: 0,  
            2: 1,  
            3: 1,  
            4: 1,  
            5: 2,  
            6: 3,  
            7: 3,  
            8: 3,  
            9: 4,  
            10: 5,  
            11: 5,  
            12: 5,  
            13: 6,  
            14: 6,  
            15: 8,  
            16: 8,
            17: 9, 
            18: 11, 
            19: 12,
        }

        default_lines_to_bus = {
            0: 1,  
            1: 4,  
            2: 2,  
            3: 3,  
            4: 4,  
            5: 3,  
            6: 4,  
            7: 6,  
            8: 8,  
            9: 5,  
            10: 10,  
            11: 11,  
            12: 12,  
            13: 7,  
            14: 8,  
            15: 9,  
            16: 13,
            17: 10, 
            18: 12, 
            19: 13,
        }


        feed_dict = {}

        # Define discount and learning rate
        feed_dict[self.learning_rate] = learning_rate
        feed_dict[self.discount] = discount        

        # Copy the latest training iteration of the model, to start where it stopped
        starting_point = copy.copy(self.current_train_iter)

        # Training loop
        for i in tqdm(range(starting_point, starting_point+max_iter)):

            # Store current training step, so that it's always up to date
            self.current_train_iter = i

            # Sample minibatch
            minibatch_indices = np.random.choice(data['_N_loads_p'].shape[0], minibatch_size)

            # From this point, the code is specific to the case14 IEEE standard power grid
            topo_state = {}
            topo_state['gens'] = np.random.binomial(1, p_topo, [minibatch_size, 5])
            topo_state['buses'] = np.random.binomial(1, p_topo, [minibatch_size, 14])
            topo_state['lines_from'] = np.random.binomial(1, p_topo, [minibatch_size, 20])
            topo_state['lines_to'] = np.random.binomial(1, p_topo, [minibatch_size, 20])

            
            feed_dict[self.buses['Pd']] = np.zeros([minibatch_size, 28])
            feed_dict[self.buses['Qd']] = np.zeros([minibatch_size, 28])
            feed_dict[self.gens['Pg']] = np.zeros([minibatch_size, 5])
            feed_dict[self.gens['Vg']] = np.zeros([minibatch_size, 5])
            feed_dict[self.gens['bus']] = np.zeros([minibatch_size, 5])
            feed_dict[self.lines['from']] = np.zeros([minibatch_size, 20])
            feed_dict[self.lines['to']] = np.zeros([minibatch_size, 20])

            for name, index in name_to_index_load.items():
                feed_dict[self.buses['Pd']][:,index] = (1-topo_state['buses'][:,index])*data['_N_loads_p'][name].values[minibatch_indices] * np.random.uniform(1.0-sigma_inj, 1.0+sigma_inj)
                feed_dict[self.buses['Pd']][:,index+14] = topo_state['buses'][:,index]*data['_N_loads_p'][name].values[minibatch_indices] * np.random.uniform(1.0-sigma_inj, 1.0+sigma_inj)
                feed_dict[self.buses['Qd']][:,index] = (1-topo_state['buses'][:,index])*data['_N_loads_q'][name].values[minibatch_indices] * np.random.uniform(1.0-sigma_inj, 1.0+sigma_inj)
                feed_dict[self.buses['Qd']][:,index+14] = topo_state['buses'][:,index]*data['_N_loads_q'][name].values[minibatch_indices] * np.random.uniform(1.0-sigma_inj, 1.0+sigma_inj)
                
            for name, index in name_to_index_prod.items():
                feed_dict[self.gens['Pg']][:,index] = data['_N_prods_p'][name].values[minibatch_indices] * np.random.uniform(1.0-sigma_inj, 1.0+sigma_inj)
                feed_dict[self.gens['Vg']][:,index] = data['_N_prods_v'][name].values[minibatch_indices]/vn_kv_prod[name] * np.random.uniform(1.0-sigma_inj, 1.0+sigma_inj)
                feed_dict[self.gens['bus']][:,index] = default_gens_bus[index] + 14* topo_state['gens'][:,index]

            for index in range(20):
                feed_dict[self.lines['from']][:,index] = default_lines_from_bus[index] + 14* topo_state['lines_from'][:,index]
                feed_dict[self.lines['to']][:,index] = default_lines_to_bus[index] + 14* topo_state['lines_to'][:,index]

            # Perform SGD step
            self.sess.run(self.opt_op, feed_dict=feed_dict)

            # Store final loss in a summary
            self.summary = self.sess.run(self.merged_summary_op, feed_dict=feed_dict)
            self.training_writer.add_summary(self.summary, self.current_train_iter)

            # Periodically log metrics and save model
            if ((save_step is not None) & (i % save_step == 0)) or (i == starting_point+max_iter-1):

                # Get minibatch train loss
                loss_final_train = self.sess.run(self.loss_final, feed_dict=feed_dict)

                # Log metrics
                logging.info('    Learning iteration {}'.format(i))
                logging.info('        Training loss (minibatch) : {}'.format(loss_final_train))

                # Save model
                self.save()

        # Save model at the end of training
        self.save()






