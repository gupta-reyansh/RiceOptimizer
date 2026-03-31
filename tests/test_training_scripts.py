import unittest
from unittest import mock

import torch

from finetune import plTrainHarness as FinetuneTrainHarness
from pretrain import (
    build_bigbird_config,
    plTrainHarness as PretrainTrainHarness,
    select_trainer_strategy,
)
from train_species_model import parse_args


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


class TestTrainingConfiguration(unittest.TestCase):
    def test_pretrain_uses_sparse_attention_config_for_short_sequences(self):
        config = build_bigbird_config(tokenizer_length=100, type_vocab_size=1)

        self.assertEqual(config.attention_type, "block_sparse")
        self.assertEqual(config.block_size, 32)

    def test_single_gpu_uses_auto_strategy(self):
        self.assertEqual(select_trainer_strategy(1), "auto")
        self.assertEqual(
            select_trainer_strategy(2), "ddp_find_unused_parameters_true"
        )

    def test_species_training_defaults_are_memory_safe(self):
        with mock.patch(
            "sys.argv",
            [
                "train_species_model.py",
                "--input_fasta",
                "/tmp/input.fasta",
                "--organism",
                "Fragaria vesca",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.accumulate_grad_batches, 6)


if __name__ == "__main__":
    unittest.main()
