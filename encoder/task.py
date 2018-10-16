import abc

import tensorflow as tf

from .types import SupervisedData, UnsupervisedData


class Task(abc.ABC):
    def __init__(self, name: str, output_dim: int):
        # TODO
        # This object should support additional model config or hyperparameters
        # with default values.

        # You can develop your own design. (without breaking the interface.)

        self.name = name
        self.output_dim = output_dim

    def extend_encoder_graph(self, encoder):
        graph = encoder.graph
        with graph.as_default(), tf.variable_scope(self.name) as vs:
            self._extend_encoder_graph(encoder)
            encoder.sess.run(
                tf.variables_initializer(
                    var_list=graph.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope=vs.name,
                    ),
                )
            )

    @abc.abstractmethod
    def _extend_encoder_graph(self, encoder):
        pass

    @abc.abstractmethod
    def validate_data(self, data):
        pass

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return f"{self.name}"


class SupervisedTask(Task):
    def validate_data(self, data: SupervisedData):
        x, y = data
        if len(x.shape) != 2:
            raise ValueError("Input data should be rank 2!")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Data pairs should have the same length!")
        if len(y.shape) != 2:
            raise ValueError("Output data should be rank 2!")
        if y.shape[1] != self.output_dim:
            raise ValueError(f"Invalid output data dimension: {y.shape[1]} != {self.output_dim}!")


class MultiLabelTask(SupervisedTask):

    def __init__(self, name: str, n_labels: int):
        super().__init__(name=name, output_dim=n_labels)

    @property
    def n_labels(self):
        return self.output_dim

    def _extend_encoder_graph(self, encoder):
        # TODO
        # Build a graph containing necessary operations and tensors
        # to train and predict multi-label data.

        # the labels will be np.array of shape (N, self.n_labels)
        # and np.int32 value in {0, 1}

        # the prediction should be based on the output of encoder.
        graph = encoder.graph
        L = graph.get_tensor_by_name(f"{encoder.scope}/L:0")

        latent_dim = int(L.get_shape()[1])
        dim_h1 = self.output_dim << 3
        dim_h2 = self.output_dim << 2
        dim_h3 = self.output_dim << 1
        # hidden layer 1
        W1 = tf.Variable(tf.zeros([latent_dim, dim_h1]))
        b1 = tf.Variable(tf.zeros([dim_h1]))
        h1 = tf.add(tf.matmul(L, W1), b1)
        # hidden layer 2
        W2 = tf.Variable(tf.zeros([dim_h1, dim_h2]))
        b2 = tf.Variable(tf.zeros([dim_h2]))
        h2 = tf.add(tf.matmul(h1, W2), b2)
        # output layer (latent space)
        W3 = tf.Variable(tf.zeros([dim_h2, dim_h3]))
        b3 = tf.Variable(tf.zeros([dim_h3]))
        h3 = tf.add(tf.matmul(h2, W3), b3)
        Y_pred = tf.reshape(h3, [-1, self.output_dim, 2], name="Y_pred")
        # true labels
        Y_ = tf.placeholder(tf.int32, [None, self.output_dim], name="Y_")
        Y_onehot = tf.one_hot(Y_, 2)
        # train
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=Y_pred),
            name="loss"
        )
        train_step = tf.train.RMSPropOptimizer(0.05).minimize(loss, name='train_step')


class MultiClassTask(SupervisedTask):

    def __init__(self, name: str, n_classes: int):
        super().__init__(name=name, output_dim=1)
        self.n_classes = n_classes

    def _extend_encoder_graph(self, encoder):
        # TODO
        # Build a graph containing necessary operations and tensors
        # to train and predict multi-class data.

        # the labels will be np.array of shape (N, 1)
        # and np.int32 value in [0, n_classes)

        # the prediction should be based on the output of encoder.
        graph = encoder.graph
        L = graph.get_tensor_by_name(f"{encoder.scope}/L:0")

        latent_dim = int(L.get_shape()[1])
        dim_h1 = self.n_classes << 2
        dim_h2 = self.n_classes << 1
        # hidden layer 1
        W1 = tf.Variable(tf.zeros([latent_dim, dim_h1]))
        b1 = tf.Variable(tf.zeros([dim_h1]))
        h1 = tf.add(tf.matmul(L, W1), b1)
        # hidden layer 2
        W2 = tf.Variable(tf.zeros([dim_h1, dim_h2]))
        b2 = tf.Variable(tf.zeros([dim_h2]))
        h2 = tf.add(tf.matmul(h1, W2), b2)
        # output layer (latent space)
        W3 = tf.Variable(tf.zeros([dim_h2, self.n_classes]))
        b3 = tf.Variable(tf.zeros([self.n_classes]))
        Y_pred = tf.nn.softmax(tf.matmul(h2, W3) + b3, name="Y_pred")
        # true labels
        Y_ = tf.placeholder(tf.int32, [None, 1], name="Y_")
        Y_onehot = tf.one_hot(Y_, self.n_classes)
        # train
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=Y_pred),
            name="loss"
        )
        train_step = tf.train.RMSPropOptimizer(0.05).minimize(loss, name='train_step')


class UnsupervisedTask(Task):
    def validate_data(self, data: UnsupervisedData):
        if len(data.shape) != 2:
            raise ValueError("Output data should be rank 2!")
        if data.shape[1] != self.output_dim:
            raise ValueError(
                f"Invalid output data dimension: {data.shape[1]} != {self.output_dim}!")


class AutoEncoderTask(UnsupervisedTask):

    def _extend_encoder_graph(self, encoder):
        # TODO
        # Build a graph containing necessary operations and tensors
        # to reconstruct the original input data.

        # the prediction should be based on the output of encoder.
        graph = encoder.graph
        Y = graph.get_tensor_by_name(f"{encoder.scope}/X:0")
        L = graph.get_tensor_by_name(f"{encoder.scope}/L:0")
        latent_dim = int(L.get_shape()[1])
        output_dim = int(Y.get_shape()[1])
        dim_h1 = latent_dim << 1
        dim_h2 = latent_dim << 2
        # hidden layer 1
        W1 = tf.Variable(tf.zeros([latent_dim, dim_h1]))
        b1 = tf.Variable(tf.zeros([dim_h1]))
        h1 = tf.add(tf.matmul(L, W1), b1)
        # hidden layer 2
        W2 = tf.Variable(tf.zeros([dim_h1, dim_h2]))
        b2 = tf.Variable(tf.zeros([dim_h2]))
        h2 = tf.add(tf.matmul(h1, W2), b2)
        # output layer (latent space)
        W3 = tf.Variable(tf.zeros([dim_h2, self.output_dim]))
        b3 = tf.Variable(tf.zeros([self.output_dim]))
        Y_pred = tf.add(tf.matmul(h2, W3), b3, name="Y_pred")
        # train
        loss = tf.reduce_mean(
            tf.squared_difference(Y, Y_pred),
            name="loss"
        )
        train_step = tf.train.RMSPropOptimizer(0.05).minimize(loss, name='train_step')

