import tensorflow as tf


class NDAdamOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, vec_axes={}, name="NDAdam"):

        super(NDAdamOptimizer, self).__init__(False, name)
        if isinstance(learning_rate, list) or isinstance(learning_rate, tuple):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = (learning_rate, 50 * learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.vec_axes = vec_axes

        self.m = {}
        self.u = {}
        self.t = tf.Variable(0, dtype=tf.int64, trainable=False)

        for v in tf.trainable_variables():
            v_shape = v.get_shape().as_list()
            self.m[v] = tf.Variable(tf.zeros(v_shape), trainable=False)
            if v in vec_axes:
                for i in vec_axes[v]:
                    v_shape[i] = 1
            self.u[v] = tf.Variable(tf.zeros(v_shape), trainable=False)

    def apply_gradients(self, gvs, global_step=None, name=None):
        update_ops = []
        t = self.t.assign_add(1)
        update_ops.append(t)
        if global_step is not None:
            update_ops.append(global_step.assign_add(1))
        t = tf.to_float(t)

        for (g, v) in gvs:
            if v in self.vec_axes:
                g_proj = tf.reduce_sum(g * v, self.vec_axes[v], keep_dims=True)
                g -= g_proj * v
                g2 = tf.reduce_sum(tf.square(g), self.vec_axes[v], keep_dims=True)
            else:
                g2 = tf.square(g)
            m = self.m[v].assign(self.beta1 * self.m[v] + (1 - self.beta1) * g)
            u = self.u[v].assign(self.beta2 * self.u[v] + (1 - self.beta2) * g2)
            m_hat = m / (1 - tf.pow(self.beta1, t))
            u_hat = u / (1 - tf.pow(self.beta2, t))

            if v in self.vec_axes:
                updated_v = v - self.learning_rate[1] * m_hat / (tf.sqrt(u_hat) + self.epsilon)
                v_norms = tf.sqrt(tf.reduce_sum(tf.square(updated_v), self.vec_axes[v], keep_dims=True))
                updated_v = v.assign(updated_v / v_norms)
            else:
                updated_v = v - self.learning_rate[0] * m_hat / (tf.sqrt(u_hat) + self.epsilon)
                updated_v = v.assign(updated_v)
            update_ops.append(updated_v)

        return tf.group(*update_ops)
