import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from braindecode.datasets import BaseConcatDataset, TUHAbnormal
from braindecode.models import CBraMod
from braindecode.preprocessing import create_fixed_length_windows


DEFAULT_DATASET_PATH = "/data/parietal/store2/data/tuh_eeg_abnormal"
DEFAULT_CBRAMOD_REPO = "braindecode/cbramod-pretrained"
DEFAULT_PATCH_SIZE = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract TUH Abnormal window embeddings from a Braindecode foundation model."
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-version", default="v3.0.1")
    parser.add_argument("--model", default="cbramod")
    parser.add_argument("--window-size-s", type=float, default=60.0)
    parser.add_argument("--window-stride-s", type=float, default=60.0)
    parser.add_argument("--max-recordings", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="artifacts/")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_dataset(args):
    dataset = TUHAbnormal(
        path=args.dataset_path,
        version=args.dataset_version,
        preload=False,
        target_name="pathological",
        n_jobs=1,
    )
    if args.max_recordings is None:
        return dataset
    return BaseConcatDataset(dataset.datasets[: args.max_recordings])


def get_expected_input_channels(model) -> int:
    if getattr(model, "channel_projection", None) is not None:
        return model.channel_projection.in_channels
    return model.n_chans


def match_model_channels(x: np.ndarray, expected_n_chans: int) -> np.ndarray:
    current_n_chans = x.shape[0]
    if current_n_chans == expected_n_chans:
        return x
    if current_n_chans > expected_n_chans:
        return x[:expected_n_chans]

    padded = np.zeros((expected_n_chans, x.shape[1]), dtype=x.dtype)
    padded[:current_n_chans] = x
    return padded


def align_window_samples(raw, window_size_s: float, patch_size: int) -> int:
    requested = min(raw.n_times, int(window_size_s * raw.info["sfreq"]))
    aligned = (requested // patch_size) * patch_size
    if aligned == 0:
        raise ValueError(
            f"Requested window is too short for patch size {patch_size}: got {requested} samples."
        )
    return aligned


def get_window_params(dataset: BaseConcatDataset, args):
    raw = dataset.datasets[0].raw
    window_size_samples = align_window_samples(
        raw, args.window_size_s, DEFAULT_PATCH_SIZE
    )
    stride_requested = int(args.window_stride_s * raw.info["sfreq"])
    window_stride_samples = max(DEFAULT_PATCH_SIZE, stride_requested)
    window_stride_samples = (
        window_stride_samples // DEFAULT_PATCH_SIZE
    ) * DEFAULT_PATCH_SIZE
    return len(raw.ch_names), window_size_samples, window_stride_samples


def get_model(args, n_chans: int, n_times: int):
    if args.model == "cbramod":
        return CBraMod.from_pretrained(
            args.pretrained_repo,
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=2,
            return_encoder_output=True,
            strict=False,
            local_files_only=args.local_files_only,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def create_windows(dataset: BaseConcatDataset, window_size_samples: int, window_stride_samples: int):
    return create_fixed_length_windows(
        dataset,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=True,
        preload=False,
        targets_from="metadata",
        n_jobs=1,
    )


def extract_window_embedding(model, window_x: np.ndarray, device: str) -> np.ndarray:
    x = match_model_channels(window_x, get_expected_input_channels(model))
    x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        features = model(x, return_features=True)["features"]
    embedding = features.mean(dim=(1, 2))
    return embedding.squeeze(0).detach().cpu().numpy()


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args()

    print("Loading TUH Abnormal...")
    dataset = load_dataset(args)
    print(f"Using {len(dataset.datasets)} recordings")

    n_chans, window_size_samples, window_stride_samples = get_window_params(
        dataset, args
    )
    print(
        f"Windowing with size={window_size_samples} samples "
        f"stride={window_stride_samples} samples"
    )

    print("Loading pretrained CBraMod...")
    model = get_model(args, n_chans=n_chans, n_times=window_size_samples)
    model = model.to(args.device)
    model.eval()

    print("Creating fixed-length windows...")
    windows = create_windows(dataset, window_size_samples, window_stride_samples)

    print("Extracting window embeddings...")
    patient_embeddings = []
    metadata = []
    for patient_index, patient_windows in enumerate(windows.datasets):
        window_embeddings = []
        for window_index in range(len(patient_windows)):
            window_x, _, _ = patient_windows[window_index]
            window_embeddings.append(
                extract_window_embedding(model, window_x, args.device)
            )

        patient_embedding_array = np.stack(window_embeddings)
        patient_embeddings.append(patient_embedding_array)
        metadata.append(
            {
                "patient_index": patient_index,
                "path": patient_windows.description["path"],
                "train": bool(patient_windows.description["train"]),
                "pathological": bool(patient_windows.description["pathological"]),
                "n_windows": int(patient_embedding_array.shape[0]),
                "embedding_dim": int(patient_embedding_array.shape[1]),
            }
        )
        print(
            f"{patient_index + 1}/{len(windows.datasets)} "
            f"windows={patient_embedding_array.shape[0]} "
            f"embedding_dim={patient_embedding_array.shape[1]}"
        )

    min_windows = min(embedding.shape[0] for embedding in patient_embeddings)
    embedding_array = np.stack(
        [embedding[:min_windows] for embedding in patient_embeddings]
    )

    output_dir = Path(args.output_dir) / f"{args.model}_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embedding_array)
    with open(output_dir / "metadata.json", "w", encoding="ascii") as handle:
        json.dump(metadata, handle, indent=2)
    with open(output_dir / "args.json", "w", encoding="ascii") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"Saved embeddings to {output_dir / 'embeddings.npy'}")
    print(f"Saved metadata to {output_dir / 'metadata.json'}")
    print(f"Embedding shape: {embedding_array.shape}")
    print(f"Kept {min_windows} windows per patient")


if __name__ == "__main__":
    main()
