import pytest
import tempfile
import shutil

import numpy as np

from encoder import Encoder
from encoder.model import MultiTaskModel
from encoder.task import (
    MultiLabelTask,
    MultiClassTask,
    AutoEncoderTask,
)


def fake_multi_label_mapping(x, n_labels):
    return (x[:, 0: n_labels] > 0.5).astype(np.int32)


def fake_multi_class_mapping(x, n_class):
    return np.expand_dims(np.argmax(x, axis=1) % n_class, axis=-1)


class TestMultiTaskEncoder:
    input_dim = 200
    latent_dim = 20

    @pytest.fixture(scope="class")
    def multi_task_model(self):
        encoder = Encoder(
            input_dim=self.input_dim,
            output_dim=self.latent_dim,
        )
        return MultiTaskModel(encoder=encoder)

    @pytest.fixture(scope="class")
    def supervised_tasks_and_data(self):
        mlt = MultiLabelTask('multi_label_test', 5)
        mct = MultiClassTask('multi_class_test', 10)

        x = np.random.randn(40, self.input_dim)
        x_mlt = x[:30]
        x_mct = x[30:]
        mlt_data = x_mlt, fake_multi_label_mapping(x_mlt, mlt.n_labels)
        mct_data = x_mct, fake_multi_class_mapping(x_mct, mct.n_classes)

        return {
            mlt: mlt_data,
            mct: mct_data,
        }

    @pytest.fixture(scope="class")
    def unsupervised_tasks_and_data(self):
        x = np.random.randn(100, self.input_dim)
        aet = AutoEncoderTask('auto_encoder_test', self.input_dim)
        aet_data = x
        return {aet: aet_data}

    @pytest.fixture(scope="class")
    def tasks_and_data(self, supervised_tasks_and_data, unsupervised_tasks_and_data):
        tasks = supervised_tasks_and_data.copy()
        tasks.update(unsupervised_tasks_and_data)
        return tasks

    def test_add_task(self, multi_task_model, tasks_and_data):
        for task in tasks_and_data.keys():
            multi_task_model.add_task(task)
        with pytest.raises(RuntimeError):
            multi_task_model.add_task(task)

    def test_evaluate(self, multi_task_model, tasks_and_data):
        for task, data in tasks_and_data.items():
            loss = multi_task_model.evaluate(task, data)
            assert loss.shape == ()

    def evaluate_on_tasks(self, multi_task_model, tasks_and_data):
        return [
            multi_task_model.evaluate(task, data)
            for task, data in tasks_and_data.items()
        ]

    def test_fit(
            self,
            multi_task_model,
            supervised_tasks_and_data,
            unsupervised_tasks_and_data,
            tasks_and_data,
        ):
        original_losses = self.evaluate_on_tasks(multi_task_model, tasks_and_data)
        multi_task_model.fit(
            supervised_data=supervised_tasks_and_data,
            unsupervised_data=unsupervised_tasks_and_data,
        )
        new_losses = self.evaluate_on_tasks(multi_task_model, tasks_and_data)
        assert all([
            new_loss < original_loss
            for new_loss, original_loss in zip(new_losses, original_losses)
        ])

    def test_encoder_encode(self, multi_task_model):
        encoder = multi_task_model.encoder
        x = np.random.randn(100, self.input_dim)
        y = encoder.encode(x)
        assert y.shape == (100, self.latent_dim)

    def test_encoder_save_load(self, multi_task_model):
        encoder = multi_task_model.encoder
        path = tempfile.mkdtemp()

        encoder.save(path=path)
        loaded = Encoder.load(path=path)

        x = np.random.randn(100, self.input_dim)
        assert encoder.input_dim == loaded.input_dim
        assert encoder.output_dim == loaded.output_dim
        np.testing.assert_array_almost_equal(
            encoder.encode(x),
            loaded.encode(x),
            decimal=6,
        )
        shutil.rmtree(path)

    # new test
    def test_encoder_save_load_2(self, multi_task_model):
        encoder = multi_task_model.encoder
        path1 = tempfile.mkdtemp()
        path2 = tempfile.mkdtemp()

        encoder.save(path=path1)
        loaded1 = Encoder.load(path=path1)
        loaded1.save(path=path2)
        loaded2 = Encoder.load(path=path2)

        x = np.random.randn(100, self.input_dim)
        assert encoder.input_dim == loaded2.input_dim
        assert encoder.output_dim == loaded2.output_dim
        np.testing.assert_array_almost_equal(
            encoder.encode(x),
            loaded2.encode(x),
            decimal=6,
        )
        shutil.rmtree(path1)
        shutil.rmtree(path2)

    # def test_fit_2(
    #         self,
    #         multi_task_model,
    #         supervised_tasks_and_data,
    #         unsupervised_tasks_and_data,
    #         tasks_and_data,
    #     ):
    #     n_step = 1
    #     original_losses = self.evaluate_on_tasks(multi_task_model, tasks_and_data)
    #     print(f'original_losses: {original_losses}')
    #     for i in range(n_step):
    #         multi_task_model.fit(
    #             supervised_data=supervised_tasks_and_data,
    #             unsupervised_data=unsupervised_tasks_and_data,
    #         )
    #         temp_losses = self.evaluate_on_tasks(multi_task_model, tasks_and_data)
    #         print(f'step_{i}_losses:{temp_losses}')
    #     new_losses = self.evaluate_on_tasks(multi_task_model, tasks_and_data)
    #     print(f'new_losses: {new_losses}')

    #     graph = multi_task_model.encoder.graph
    #     sess = multi_task_model.encoder.sess
    #     X = graph.get_tensor_by_name("Encoder/X:0")
    #     L = graph.get_tensor_by_name(f"Encoder/L:0")
    #     Y_pred = graph.get_tensor_by_name("auto_encoder_test/Y_pred:0")
    #     for task, data in unsupervised_tasks_and_data.items():
    #         print("X")
    #         print(data[:5, :4])
    #         print("Y_pred")
    #         print(sess.run(Y_pred, feed_dict={X: data})[:5, :4])
    #         print("L")
    #         print(sess.run(L, feed_dict={X: data})[:5, :4])
    #     W = graph.get_tensor_by_name("auto_encoder_test/W:0")
    #     b = graph.get_tensor_by_name("auto_encoder_test/b:0")
    #     print("W")
    #     print(sess.run(W)[:4, :4])
    #     print("b")
    #     print(sess.run(b)[:4])

    #     assert all([
    #         new_loss < original_loss
    #         for new_loss, original_loss in zip(new_losses, original_losses)
    #     ])
