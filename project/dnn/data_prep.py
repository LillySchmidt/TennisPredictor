import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dnn.config import data_cfg, paths  # noqa: E402
from dnn.utils import ensure_dir, save_json, seed_everything  # noqa: E402
from ML import data_prep as ml_dp  # noqa: E402
from ML import preprocess as ml_pp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tennis matches data for DNN training.")
    parser.add_argument("--raw", type=Path, default=data_cfg.raw_csv, help="Path to raw all_matches.csv")
    parser.add_argument("--out", type=Path, default=data_cfg.processed_npz, help="Output npz path for processed data")
    parser.add_argument("--test_size", type=float, default=data_cfg.test_size, help="Test split size")
    parser.add_argument("--val_size", type=float, default=data_cfg.val_size, help="Validation split size (fraction of train)")
    parser.add_argument("--random_state", type=int, default=data_cfg.random_state, help="Random seed")
    return parser.parse_args()


def _to_dense(arr):
    if hasattr(arr, "toarray"):
        return arr.toarray()
    return arr


def main() -> None:
    args = parse_args()
    seed_everything(args.random_state)

    raw_path = args.raw
    ensure_dir(paths.data_processed)
    ensure_dir(paths.artifacts)
    ensure_dir(paths.dnn_models)
    ensure_dir(paths.dnn_reports)
    ensure_dir(paths.dnn_images)

    prematch = ml_dp.build_prematch_frame(raw_path, seed=args.random_state, clean=True)
    y_cls = prematch["label"].astype(int)
    X = prematch.drop(columns=["label", "match_id"])

    # Split data
    X_train, X_temp, y_cls_train, y_cls_temp = train_test_split(
        X,
        y_cls,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_cls,
    )
    val_size_adj = args.val_size / (1 - args.test_size)
    X_val, X_test, y_cls_val, y_cls_test = train_test_split(
        X_temp,
        y_cls_temp,
        test_size=1 - val_size_adj,
        random_state=args.random_state,
        stratify=y_cls_temp,
    )

    preprocessor, numeric_cols, cat_cols = ml_pp.build_preprocessor(X_train)
    X_train_t = _to_dense(preprocessor.fit_transform(X_train))
    X_val_t = _to_dense(preprocessor.transform(X_val))
    X_test_t = _to_dense(preprocessor.transform(X_test))

    # Save arrays
    out_path = args.out
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        X_train=X_train_t.astype(np.float32),
        X_val=X_val_t.astype(np.float32),
        X_test=X_test_t.astype(np.float32),
        cls_train=y_cls_train.to_numpy().astype(np.float32),
        cls_val=y_cls_val.to_numpy().astype(np.float32),
        cls_test=y_cls_test.to_numpy().astype(np.float32),
    )

    # Save processor and metadata
    processor_path = paths.artifacts / "processor.joblib"
    joblib.dump(preprocessor, processor_path)
    meta = {
        "feature_columns": list(X.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "shapes": {
            "X_train": X_train_t.shape,
            "X_val": X_val_t.shape,
            "X_test": X_test_t.shape,
        },
        "splits": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "random_state": args.random_state,
        "source": "ML.build_prematch_frame",
    }
    save_json(meta, paths.artifacts / "metadata.json")
    print(f"Saved processed arrays to {out_path}")
    print(f"Processor saved to {processor_path}")


if __name__ == "__main__":
    main()
