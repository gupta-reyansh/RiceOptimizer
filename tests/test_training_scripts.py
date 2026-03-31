import unittest

import torch

from finetune import plTrainHarness as FinetuneTrainHarness
from pretrain import plTrainHarness as PretrainTrainHarness


class DummyTrainer:
    def __init__(self, estimated_stepping_batches):
        self.estimated_stepping_batches = estimated_stepping_batches


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)


class TestTrainingHarnessSchedulers(unittest.TestCase):
    def test_pretrain_harness_skips_one_cycle_for_iterable_loader(self):
        harness = PretrainTrainHarness(DummyModel(), 1e-3, 0.1)
        harness._trainer = DummyTrainer(-1)

        optimizer = harness.configure_optimizers()

        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_finetune_harness_skips_one_cycle_for_iterable_loader(self):
        harness = FinetuneTrainHarness(DummyModel(), 1e-3, 0.1)
        harness._trainer = DummyTrainer(-1)

        optimizer = harness.configure_optimizers()

        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_pretrain_harness_keeps_scheduler_for_known_total_steps(self):
        harness = PretrainTrainHarness(DummyModel(), 1e-3, 0.1)
        harness._trainer = DummyTrainer(12)

        optimizers, schedulers = harness.configure_optimizers()

        self.assertEqual(len(optimizers), 1)
        self.assertEqual(len(schedulers), 1)
        self.assertEqual(schedulers[0]["scheduler"].total_steps, 12)


if __name__ == "__main__":
    unittest.main()
