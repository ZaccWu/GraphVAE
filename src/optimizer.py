import tensorflow as tf

class OptimizerAE(object):
    """ Optimizer for non-variational autoencoders """
    def __init__(self, params, preds, labels, posWeight, norm):
        param = params
        predsSub = preds
        labelsSub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = predsSub,
                                                     targets = labelsSub,
                                                     pos_weight = posWeight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = param['learning_rate'])
        self.optOp = self.optimizer.minimize(self.cost)
        self.gradsVars = self.optimizer.compute_gradients(self.cost)
        self.correctPrediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(predsSub), 0.5), tf.int32),
                     tf.cast(labelsSub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))

class OptimizerVAE(object):
    """ Optimizer for variational autoencoders """
    def __init__(self, params, preds, labels, model, numNodes, posWeight, norm):
        param = params
        predsSub = preds
        labelsSub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = predsSub,
                                                     targets = labelsSub,
                                                     pos_weight = posWeight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = param['learning_rate'])
        # Latent loss
        self.logLik = self.cost # What's this for?
        self.kl = (0.5 / numNodes) * tf.reduce_mean(tf.reduce_sum(1+2*model.zLogStd-tf.square(model.zMean)-tf.square(tf.exp(model.zLogStd)), 1))
        self.cost -= self.kl
        self.optOp = self.optimizer.minimize(self.cost)
        self.gradsVars = self.optimizer.compute_gradients(self.cost)
        self.correctPrediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(predsSub), 0.5), tf.int32),
                                              tf.cast(labelsSub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))