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
        self.lr = 1e-2

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

    def create_model_dims(self, input_dim, output_dim):
        max_hid_layer = 2
        dim_list = [input_dim, output_dim]
        hid_layer = 0
        layer_size = 32
        while layer_size < (input_dim << 1) and hid_layer < max_hid_layer:
            if layer_size > self.output_dim:
                dim_list.insert(1, layer_size)
                hid_layer += 1
            layer_size <<= 1
        return dim_list


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
        scope = encoder.scope
        h = graph.get_tensor_by_name(f"{scope}/L:0")
        keep_prob = graph.get_tensor_by_name(f"{scope}/keep_prob:0")

        # model
        dim_list = self.create_model_dims(encoder.output_dim, self.output_dim << 1)
        for i in range(1, len(dim_list)):
            W = tf.Variable(tf.random_normal([dim_list[i - 1], dim_list[i]]), name="W")
            b = tf.Variable(tf.zeros([dim_list[i]]), name="b")
            if len(dim_list) - (i + 1):
                h = tf.sigmoid(tf.add(tf.matmul(h, W), b))
                h = tf.nn.dropout(h, keep_prob)
        h = tf.add(tf.matmul(h, W), 3)
        Y_pred = tf.reshape(h, [-1, self.output_dim, 2], name="Y_pred")
        # true labels
        Y_ = tf.placeholder(tf.int32, [None, self.output_dim], name="Y_")
        Y_onehot = tf.one_hot(Y_, 2)
        # train
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=Y_pred),
            name="loss"
        )
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        train_step = tf.train.RMSPropOptimizer(self.lr).minimize(
            loss, var_list=train_vars, name='train_step'
        )


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
        scope = encoder.scope
        h = graph.get_tensor_by_name(f"{scope}/L:0")
        keep_prob = graph.get_tensor_by_name(f"{scope}/keep_prob:0")

        # model
        dim_list = self.create_model_dims(encoder.output_dim, self.n_classes)
        for i in range(1, len(dim_list)):
            W = tf.Variable(tf.random_normal([dim_list[i - 1], dim_list[i]]), name="W")
            b = tf.Variable(tf.zeros([dim_list[i]]), name="b")
            if len(dim_list) - (i + 1):
                h = tf.sigmoid(tf.add(tf.matmul(h, W), b))
                h = tf.nn.dropout(h, keep_prob)
        Y_pred = tf.add(tf.matmul(h, W), b, name="Y_pred")
        # true labels
        Y_ = tf.placeholder(tf.int32, [None, 1], name="Y_")
        Y_onehot = tf.squeeze(tf.one_hot(Y_, self.n_classes), [1], name="Y_onehot")
        # train
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=Y_pred),
            name="loss"
        )
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        train_step = tf.train.RMSPropOptimizer(self.lr).minimize(
            loss, var_list=train_vars, name='train_step'
        )


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
        scope = encoder.scope
        Y = graph.get_tensor_by_name(f"{scope}/X:0")
        h = graph.get_tensor_by_name(f"{scope}/L:0")
        keep_prob = graph.get_tensor_by_name(f"{scope}/keep_prob:0")

        dim_list = encoder.model_dims[::-1]
        for i in range(1, len(dim_list)):
            W = tf.Variable(tf.random_normal([dim_list[i - 1], dim_list[i]]), name="W")
            b = tf.Variable(tf.zeros([dim_list[i]]), name="b")
            if len(dim_list) - (i + 1):
                h = tf.sigmoid(tf.add(tf.matmul(h, W), b))
                h = tf.nn.dropout(h, keep_prob)
        Y_pred = tf.add(tf.matmul(h, W), b, name="Y_pred")

        # train
        loss = tf.reduce_mean(
            tf.squared_difference(Y, Y_pred),
            name="loss"
        )
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        train_step = tf.train.RMSPropOptimizer(self.lr).minimize(
            loss, var_list=train_vars, name='train_step'
        )
