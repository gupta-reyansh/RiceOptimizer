"""
Train a supported CodonTransformer species model directly from a CDS FASTA file.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from CodonTransformer.CodonData import prepare_training_data, read_fasta_file
from CodonTransformer.CodonEvaluation import get_CSI_value, get_CSI_weights
from CodonTransformer.CodonUtils import FINE_TUNE_ORGANISMS, ORGANISM2ID


REPO_ROOT = Path(__file__).resolve().parent


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug or "species"


def resolve_work_dir(work_dir: str, organism: str) -> Path:
    if work_dir:
        return Path(work_dir).expanduser().resolve()
    return (REPO_ROOT / "workflows" / slugify(organism)).resolve()


def validate_top_fraction(top_fraction: float) -> float:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be greater than 0 and less than or equal to 1.")
    return top_fraction


def ensure_supported_organism(organism: str) -> None:
    if organism not in ORGANISM2ID:
        raise ValueError(
            f"Unsupported organism: {organism}. "
            "Only organisms already present in ORGANISM2ID can be finetuned."
        )

    if organism not in FINE_TUNE_ORGANISMS:
        print(
            "Warning: the organism is present in ORGANISM2ID but is not listed in "
            "FINE_TUNE_ORGANISMS. Training is still attempted, but it is outside the "
            "published finetuning examples."
        )


def prepare_finetune_inputs(
    input_fasta: str,
    organism: str,
    work_dir: Path,
    top_fraction: float,
    keep_all_records: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path]:
    raw_dir = work_dir / "data" / "raw"
    processed_dir = work_dir / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    parsed_records_path = raw_dir / "parsed_cds_records.csv"
    scored_records_path = raw_dir / "scored_cds_records.csv"
    training_csv_path = raw_dir / "training_sequences.csv"
    training_json_path = processed_dir / "training_data.json"

    parsed_df = read_fasta_file(
        input_file=str(Path(input_fasta).expanduser().resolve()),
        save_to_file=None,
        organism=organism,
    )

    if parsed_df.empty:
        raise ValueError("No FASTA records were parsed from the input file.")

    parsed_df.to_csv(parsed_records_path, index=False)

    training_candidates = parsed_df.copy()
    if not keep_all_records:
        training_candidates = training_candidates.loc[
            training_candidates["correct_seq"]
        ].copy()

    training_candidates = training_candidates.loc[
        training_candidates["dna"].str.len() % 3 == 0
    ].copy()

    if training_candidates.empty:
        raise ValueError(
            "No eligible CDS records remained after filtering. "
            "Consider checking sequence validity or using --keep_all_records."
        )

    weights = get_CSI_weights(training_candidates["dna"].tolist())
    training_candidates["CSI"] = training_candidates["dna"].map(
        lambda seq: get_CSI_value(seq, weights)
    )
    training_candidates = training_candidates.sort_values(
        "CSI", ascending=False
    ).reset_index(drop=True)
    training_candidates.to_csv(scored_records_path, index=False)

    top_n = max(1, int(len(training_candidates) * top_fraction))
    selected_df = training_candidates.head(top_n).copy()
    training_df = selected_df.loc[:, ["dna", "protein"]].copy()
    training_df["organism"] = organism
    training_df.to_csv(training_csv_path, index=False)

    prepare_training_data(
        dataset=str(training_csv_path),
        output_file=str(training_json_path),
    )

    return (
        parsed_df,
        training_candidates,
        training_df,
        training_csv_path,
        training_json_path,
    )


def run_finetune(training_json_path: Path, checkpoint_dir: Path, args: argparse.Namespace) -> None:
    from finetune import main as finetune_main

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    finetune_args = argparse.Namespace(
        dataset_dir=str(training_json_path),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_filename=args.checkpoint_filename,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_gpus=args.num_gpus,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        save_every_n_steps=args.save_every_n_steps,
        seed=args.seed,
        debug=args.debug,
    )
    finetune_main(finetune_args)


def write_run_metadata(
    work_dir: Path,
    organism: str,
    input_fasta: str,
    parsed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    training_df: pd.DataFrame,
    training_csv_path: Path,
    training_json_path: Path,
    args: argparse.Namespace,
) -> Path:
    metadata_path = work_dir / "run_metadata.json"
    payload = {
        "input_fasta": str(Path(input_fasta).expanduser().resolve()),
        "organism": organism,
        "work_dir": str(work_dir),
        "parsed_records": len(parsed_df),
        "eligible_records": len(candidate_df),
        "selected_records": len(training_df),
        "top_fraction": args.top_fraction,
        "keep_all_records": args.keep_all_records,
        "training_csv": str(training_csv_path),
        "training_json": str(training_json_path),
        "checkpoint_dir": str((work_dir / "checkpoints").resolve()),
        "checkpoint_filename": args.checkpoint_filename,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-species CodonTransformer finetuning pipeline from a CDS FASTA file."
    )
    parser.add_argument(
        "--input_fasta",
        type=str,
        required=True,
        help="Path to the CDS FASTA file.",
    )
    parser.add_argument(
        "--organism",
        type=str,
        required=True,
        help="Supported organism name from ORGANISM2ID.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="",
        help="Working directory for generated files. Defaults to <repo_root>/workflows/<organism_slug>.",
    )
    parser.add_argument(
        "--top_fraction",
        type=float,
        default=0.1,
        help="Top CSI fraction to retain for finetuning. Use 1.0 to keep all eligible records.",
    )
    parser.add_argument(
        "--keep_all_records",
        action="store_true",
        help="Keep records flagged as incorrect_seq when building the candidate pool.",
    )
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="finetune.ckpt",
        help="Checkpoint filename passed to finetune.py.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for training."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=15, help="Maximum number of epochs to train."
    )
    parser.add_argument(
        "--num_workers", type=int, default=5, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients.",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps to use for warmup.",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=512,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable finetune.py debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.top_fraction = validate_top_fraction(args.top_fraction)
    ensure_supported_organism(args.organism)

    work_dir = resolve_work_dir(args.work_dir, args.organism)
    checkpoint_dir = work_dir / "checkpoints"
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)

    parsed_df, candidate_df, training_df, training_csv_path, training_json_path = (
        prepare_finetune_inputs(
            input_fasta=args.input_fasta,
            organism=args.organism,
            work_dir=work_dir,
            top_fraction=args.top_fraction,
            keep_all_records=args.keep_all_records,
        )
    )

    metadata_path = write_run_metadata(
        work_dir=work_dir,
        organism=args.organism,
        input_fasta=args.input_fasta,
        parsed_df=parsed_df,
        candidate_df=candidate_df,
        training_df=training_df,
        training_csv_path=training_csv_path,
        training_json_path=training_json_path,
        args=args,
    )

    print(f"Parsed FASTA records: {len(parsed_df)}")
    print(f"Eligible records after filtering: {len(candidate_df)}")
    print(f"Selected records for finetuning: {len(training_df)}")
    print(f"Training CSV: {training_csv_path}")
    print(f"Training JSON: {training_json_path}")
    print(f"Run metadata: {metadata_path}")

    run_finetune(
        training_json_path=training_json_path,
        checkpoint_dir=checkpoint_dir,
        args=args,
    )

    print(f"Finetuning completed. Checkpoints are available in {checkpoint_dir}")


if __name__ == "__main__":
    main()
