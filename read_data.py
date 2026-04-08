from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from braindecode.datasets import TUHAbnormal
from braindecode.models import BENDR, CBraMod, EEGPT, REVE
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader


@dataclass
class PipelineConfig:
    data_root: str = "/data/parietal/store2/data/tuh_eeg_abnormal"
    preload: bool = False
    n_jobs: int = 20
    recording_ids: Sequence[int] = (0, 1)

    window_length_seconds: int = 30
    window_stride_seconds: int = 30

    batch_size: int = 64
    num_workers: int = 20

    encoder_name: str = "CBraMod"  # Supported: CBraMod, BENDR, REVE, EEGPT
    output_features_path: str = "feature_matrix.npy"
    output_labels_path: str = "labels.npy"


def zscore_and_clip(batch_x: torch.Tensor, clip_value: float = 15.0) -> torch.Tensor:
    """Normalize each window then clip outliers (required by REVE)."""
    mean = batch_x.mean(dim=-1, keepdim=True)
    std = batch_x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return ((batch_x - mean) / std).clamp(min=-clip_value, max=clip_value)


def embeddings_to_feature_matrix(embeddings: np.ndarray, encoder_name: str) -> np.ndarray:
    """Convert raw model outputs to a 2D feature matrix."""
    if encoder_name == "CBraMod" and embeddings.ndim >= 3:
        return embeddings.mean(axis=2).reshape(embeddings.shape[0], -1)
    if encoder_name == "BENDR" and embeddings.ndim >= 3:
        return embeddings.mean(axis=-1)
    if encoder_name == "EEGPT" and embeddings.ndim == 3:
        return embeddings.mean(axis=1)
    return embeddings.reshape(embeddings.shape[0], -1)


def define_encoder(
    encoder_name: str,
    n_chans: int,
    sfreq: float,
    input_window_seconds: int,
    channel_names: Sequence[str],
) -> torch.nn.Module:
    """Load a pretrained encoder and freeze its parameters."""
    if encoder_name == "CBraMod":
        encoder = CBraMod.from_pretrained(
            "braindecode/cbramod-pretrained",
            return_encoder_output=True,
            n_chans=36,
        )
    elif encoder_name == "BENDR":
        encoder = BENDR.from_pretrained(
            "braindecode/braindecode-bendr",
            n_chans=n_chans,
            n_outputs=1,
        )
    elif encoder_name == "REVE":
        encoder = REVE.from_pretrained(
            "brain-bzh/reve-base",
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
            n_chans=n_chans,
            n_outputs=1,
        )
    elif encoder_name == "EEGPT":
        chs_info = [{"ch_name": name} for name in channel_names]
        encoder = EEGPT.from_pretrained(
            "braindecode/eegpt-pretrained",
            n_chans=n_chans,
            n_times=512,
            chs_info=chs_info,
            n_outputs=None,
            return_encoder_output=True,
        )
    else:
        raise ValueError(f"Unknown encoder name: {encoder_name}")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    num_params = sum(param.numel() for param in encoder.parameters())
    print(f"Loaded {encoder_name} with {num_params:,} parameters")
    return encoder


def compute_embeddings(
    encoder: torch.nn.Module,
    eeg_windows,
    encoder_name: str,
    batch_size: int,
    num_workers: int,
    channel_names: Sequence[str],
) -> np.ndarray:
    """Run windowed EEG data through the encoder and collect embeddings."""
    data_loader = DataLoader(
        eeg_windows,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for batch_x, _ in data_loader:
            if encoder_name == "BENDR":
                emb = encoder.encoder(batch_x)
            elif encoder_name == "REVE":
                eeg_norm = zscore_and_clip(batch_x)
                pos = encoder.get_positions(channel_names)
                pos = pos.unsqueeze(0).expand(batch_x.shape[0], -1, -1)
                layer_outputs = encoder(eeg_norm, pos, return_output=True)
                emb = layer_outputs[-1].mean(dim=1)
            else:
                ipdb.set_trace()
                emb = encoder(batch_x)

            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            if not torch.is_tensor(emb):
                emb = torch.as_tensor(emb)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def build_windows_dataset(config: PipelineConfig):
    """Load TUHAbnormal and convert recordings into fixed-length windows."""
    ds = TUHAbnormal(
        path=config.data_root,
        recording_ids=list(config.recording_ids),
        target_name="pathological",
        preload=config.preload,
        n_jobs=config.n_jobs,
    )

    sfreq = ds.datasets[0].raw.info["sfreq"]
    window_size_samples = int(sfreq * config.window_length_seconds)
    stride_samples = int(sfreq * config.window_stride_seconds)

    windows_ds = create_fixed_length_windows(
        ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size_samples,
        window_stride_samples=stride_samples,
        drop_last_window=True,
        preload=config.preload,
    )
    return windows_ds, sfreq


def main() -> None:
    config = PipelineConfig()

    windows_ds, sfreq = build_windows_dataset(config)
    raw_info = windows_ds.datasets[0].raw.info
    channel_names = raw_info["ch_names"]
    n_chans = len(channel_names)

    encoder = define_encoder(
        encoder_name=config.encoder_name,
        n_chans=n_chans,
        sfreq=sfreq,
        input_window_seconds=config.window_length_seconds,
        channel_names=channel_names,
    )

    embeddings = compute_embeddings(
        encoder=encoder,
        eeg_windows=windows_ds,
        encoder_name=config.encoder_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        channel_names=channel_names,
    )
    feature_matrix = embeddings_to_feature_matrix(embeddings, config.encoder_name)
    labels = windows_ds.description["pathological"].values

    print(f"Feature matrix shape: {feature_matrix.shape}")
    np.save(config.output_features_path, feature_matrix)
    np.save(config.output_labels_path, labels)
    print(
        f"Saved features to {config.output_features_path} and labels to {config.output_labels_path}"
    )


if __name__ == "__main__":
    main()


