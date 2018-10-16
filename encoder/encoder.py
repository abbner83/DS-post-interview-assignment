import numpy as np
import tensorflow as tf

from .types import Data

_SCOPE_NAME = "Encoder"


class Encoder:
    def __init__(self, input_dim: int, output_dim: int):
        # TODO
        # This object should support additional model config or hyperparameters
        # with default values.

        # You can develop your own design. (without breaking the interface.)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scope = _SCOPE_NAME
        self._set_up()

    def _set_up(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.variable_scope(_SCOPE_NAME):
            self._build_graph()
        self.sess = tf.Session(graph=self.graph)
        self.saver = tf.train.Saver(
            var_list=self.graph.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=_SCOPE_NAME),
        )
        self.sess.run(
            tf.variables_initializer(
                var_list=self.graph.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=_SCOPE_NAME,
                ),
            )
        )

    def _build_graph(self):
        # TODO
        # Build a graph containing necessary operations and tensors
        # to map input data to output space.

        # the inputs will be np.array of shape (N, self.input_dim), dtype: np.float32
        # outputs will be np.array of shape (N, self.output_dim), dtype: np.float32
        dim_h1 = self.output_dim << 2
        dim_h2 = self.output_dim << 1

        X = tf.placeholder(tf.float32, [None, self.input_dim], name="X")
        # hidden layer 1
        W1 = tf.Variable(tf.zeros([self.input_dim, dim_h1]))
        b1 = tf.Variable(tf.zeros([dim_h1]))
        h1 = tf.add(tf.matmul(X, W1), b1)
        # hidden layer 2
        W2 = tf.Variable(tf.zeros([dim_h1, dim_h2]))
        b2 = tf.Variable(tf.zeros([dim_h2]))
        h2 = tf.add(tf.matmul(h1, W2), b2)
        # output layer (latent space)
        W3 = tf.Variable(tf.zeros([dim_h2, self.output_dim]))
        b3 = tf.Variable(tf.zeros([self.output_dim]))
        L = tf.add(tf.matmul(h2, W3), b3, name="L")

    def encode(self, X: np.ndarray) -> np.ndarray:
        self.validate_data(X)
        # TODO
        # Return the encoded vector of X,
        # should be np.array of shape (N, self.output_dim).
        model_X = self.graph.get_tensor_by_name(f"{_SCOPE_NAME}/X:0")
        model_L = self.graph.get_tensor_by_name(f"{_SCOPE_NAME}/L:0")
        return self.sess.run(model_L, feed_dict={model_X: X})

    def validate_data(self, data: Data):
        x = data[0] if isinstance(data, tuple) else data
        if len(x.shape) != 2:
            raise ValueError("Input data should be rank 2!")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Invalid input data dimension: {x.shape[1]} != {self.input_dim}!")

    @classmethod
    def load(cls, path: str) -> object:
        # TODO
        # Restore the Encoder object
        # from given file path which has been passed to save already.

        # Hint: make use of tf.train.Saver
        # restore
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            saver = tf.train.import_meta_graph(f'{path}.meta')
            saver.restore(sess=sess, save_path=path)
        # init
        input_dim = int(graph.get_tensor_by_name(f"{_SCOPE_NAME}/X:0").get_shape()[1])
        output_dim = int(graph.get_tensor_by_name(f"{_SCOPE_NAME}/L:0").get_shape()[1])
        encoder = cls(input_dim=input_dim, output_dim=output_dim)
        # setup
        encoder.graph = graph
        encoder.sess = sess
        encoder.saver = tf.train.Saver(
            var_list=encoder.graph.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=_SCOPE_NAME),
        )
        return encoder

    def save(self, path: str):
        # TODO
        # Save the variables and hyperparameters of Encoder to given path.

        # Hint: make use of tf.train.Saver
        self.saver.save(self.sess, path)
