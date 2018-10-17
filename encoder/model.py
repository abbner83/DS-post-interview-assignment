from typing import Dict, Union

import numpy as np

from .task import Task, SupervisedTask, UnsupervisedTask
from .types import (
    Data,
    SupervisedData,
    UnsupervisedData,
)


MultiSupervisedData = Dict[SupervisedTask, SupervisedData]
MultiUnsupervisedData = Dict[UnsupervisedTask, UnsupervisedData]
MultiTaskData = Union[MultiSupervisedData, MultiUnsupervisedData]


class MultiTaskModel:
    def __init__(self, encoder):
        # [Optional TODO]
        # This object can support additional model config or hyperparameters
        # with default values.

        # You can develop your own design. (without breaking the interface.)

        self.encoder = encoder
        self._task = set()
        self.graph = encoder.graph
        self.sess = encoder.sess

    def add_task(self, task: Task):
        if task in self._task:
            raise RuntimeError("Task already exists!")
        self._task.add(task)
        task.extend_encoder_graph(self.encoder)

    def fit(
            self,
            supervised_data: MultiSupervisedData,
            unsupervised_data: MultiUnsupervisedData,
        ):
        if not supervised_data and not unsupervised_data:
            raise RuntimeError()
        self._validate_multi_task_data(supervised_data)
        self._validate_multi_task_data(unsupervised_data)
        # TODO
        # Try to minimize the losses of each tasks
        n_epoch = 100
        iter_each_epoch_sup = 5
        iter_each_epoch_unsup = 100
        graph, sess, scope = self.graph, self.sess, self.encoder.scope

        for i in range(n_epoch):
            for task, data in supervised_data.items():
                X, Y = data
                model_X = graph.get_tensor_by_name(f"{scope}/X:0")
                model_Y_ = graph.get_tensor_by_name(f"{task}/Y_:0")
                train_step = graph.get_operation_by_name(f"{task}/train_step")
                for _ in range(iter_each_epoch_sup):
                    sess.run(train_step, feed_dict={model_X: X, model_Y_: Y})
            for task, data in unsupervised_data.items():
                model_X = graph.get_tensor_by_name(f"{scope}/X:0")
                train_step = graph.get_operation_by_name(f"{task}/train_step")
                for _ in range(iter_each_epoch_unsup):
                    sess.run(train_step, feed_dict={model_X: data})

    def _validate_multi_task_data(self, multi_task_data: MultiTaskData):
        for task, data in multi_task_data.items():
            self._validate_data(task, data)

    def _validate_data(self, task: Task, data: Data):
        if task not in self._task:
            raise KeyError(f"Unregistered task: {task}.")
        self.encoder.validate_data(data)
        task.validate_data(data)

    def evaluate(self, task: Task, data: Data) -> np.ndarray:
        self._validate_data(task, data)
        # TODO
        # Return the loss of given task on given data.
        # output should be np.array of shape (). (a.k.a scalar)
        graph, sess, scope = self.graph, self.sess, self.encoder.scope

        model_X = graph.get_tensor_by_name(f"{scope}/X:0")
        loss = graph.get_tensor_by_name(f"{task}/loss:0")
        if type(data)==tuple:
            X, Y = data
            model_Y_ = graph.get_tensor_by_name(f"{task}/Y_:0")
            return sess.run(loss, feed_dict={model_X: X, model_Y_: Y})
        else:
            return sess.run(loss, feed_dict={model_X: data})
