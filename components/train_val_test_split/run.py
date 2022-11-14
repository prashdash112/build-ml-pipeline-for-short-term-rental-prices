#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import os
import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )
    logger.info("Uploading trainval_data artifact")
    trainval.to_csv('trainval_data.csv', index=False)
    artifact1 = wandb.Artifact(
        name="trainval_data",
        type="trainval_data",
        description="splitted_dataset",
        )
    artifact1.add_file('trainval_data.csv')
    run.log_artifact(artifact1)

    logger.info("Uploading test_data artifact")
    test.to_csv('test_data.csv', index=False)
    artifact2 = wandb.Artifact(
        name="test_data",
        type="test_data",
        description="splitted_dataset",
        )
    artifact2.add_file('test_data.csv')
    run.log_artifact(artifact2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)