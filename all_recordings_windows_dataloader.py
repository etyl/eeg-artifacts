from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
import torch
import warnings

TUH_PATH = "/data/parietal/store2/data/tuh_eeg_abnormal/"

WINDOW_LEN_S = 35
WINDOW_STRIDE_S = 4
BATCH_SIZE = 32
NUM_WORKERS = 4
N_BATCHES_TO_ITERATE = 5
SEED = 42
PRELOAD = False
N_JOBS = 8


def load_all_recordings_as_windows(preload=PRELOAD, window_len_s=WINDOW_LEN_S, window_stride_s=WINDOW_STRIDE_S):
    """Load all recordings and convert them into fixed-length windows."""
    ds = TUHAbnormal(
        TUH_PATH,
        target_name="pathological",
        preload=preload,
    )

    if len(ds.datasets) == 0:
        raise RuntimeError("No recordings found in TUH Abnormal dataset.")

    sfreq = float(ds.datasets[0].raw.info["sfreq"])
    window_len_samples = int(sfreq * window_len_s)
    window_stride_samples = int(sfreq * window_stride_s)

    windows_ds = create_fixed_length_windows(
        ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_len_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=True,
        preload=preload,
        n_jobs=N_JOBS,
    )
    return windows_ds


def collate_windows_with_channel_padding(batch):
    """Collate function that pads channel dimension to batch max."""
    x_list = []
    y_list = []
    original_n_channels = []

    for sample in batch:
        x = torch.as_tensor(sample[0], dtype=torch.float32)
        y = torch.as_tensor(sample[1])
        if y.ndim > 0:
            y = y.reshape(-1)[0]

        x_list.append(x)
        y_list.append(y)
        original_n_channels.append(int(x.shape[0]))

    max_channels = max(ch for ch in original_n_channels)
    n_times = int(x_list[0].shape[1])

    padded_x_list = []
    for x in x_list:
        if int(x.shape[1]) != n_times:
            raise RuntimeError(
                f"Unexpected time dimension mismatch in batch: got {int(x.shape[1])}, expected {n_times}."
            )

        pad_channels = max_channels - int(x.shape[0])
        if pad_channels > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_channels), mode="constant", value=0.0)
        padded_x_list.append(x)

    x_batch = torch.stack(padded_x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    n_channels_batch = torch.as_tensor(original_n_channels, dtype=torch.int64)
    return x_batch, y_batch, n_channels_batch


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Loading all recordings and creating fixed-length windows...")
    windows_ds = load_all_recordings_as_windows(preload=PRELOAD)
    print("Done.")
    print("Total number of windows:", len(windows_ds))

    generator = torch.Generator()
    generator.manual_seed(SEED)

    dataloader = DataLoader(
        windows_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
        generator=generator,
        collate_fn=collate_windows_with_channel_padding,
    )

    print(f"\nIterating through {N_BATCHES_TO_ITERATE} shuffled batches...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= N_BATCHES_TO_ITERATE:
            break

        x, y, n_channels = batch
        pathological_in_batch = int((y.view(-1) > 0).sum().item())

        print(
            f"batch={batch_idx + 1} | x_shape={tuple(x.shape)} | "
            f"y_shape={tuple(y.shape)} | pathological={pathological_in_batch}/{y.numel()} | "
            f"channels(min/max)={int(n_channels.min())}/{int(n_channels.max())}"
        )


if __name__ == "__main__":
    main()
