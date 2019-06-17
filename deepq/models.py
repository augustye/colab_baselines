import tensorflow as tf
import baselines.deepq.models

def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, activation="relu", weights_stddev=0.05, **extra_kwargs):

    activation = getattr(tf.nn, activation)

    if network != "mlp":
        return baselines.deepq.models.build_q_func(network, hiddens, dueling, layer_norm)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            latent = tf.contrib.layers.flatten(input_placeholder)

            with tf.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    action_out = tf.contrib.layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None, weights_initializer=tf.keras.initializers.truncated_normal(mean=0.0, stddev=weights_stddev))
                    if layer_norm:
                        action_out = tf.contrib.layers.layer_norm(action_out, center=True, scale=True)
                    action_out = activation(action_out)
                action_scores = tf.contrib.layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = tf.contrib.layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None, weights_initializer=tf.keras.initializers.truncated_normal(mean=0.0, stddev=weights_stddev))
                        if layer_norm:
                            state_out = tf.contrib.layers.layer_norm(state_out, center=True, scale=True)
                        state_out = activation(state_out)
                    state_score = tf.contrib.layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
                
            return q_out

    return q_func_builder
