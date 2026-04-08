from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import create_fixed_length_windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TUH_PATH = "/data/parietal/store2/data/tuh_eeg_abnormal/"

def load_example_data(preload, window_len_s, n_recordings=10, recording_ids=None):
    """Create windowed dataset from subjects of the TUH Abnormal dataset.
    (Taken from the example in the doc)

    Parameters
    ----------
    preload: bool
        If True, use eager loading, otherwise use lazy loading.
    window_len_s: int
        Window length in seconds.
    n_recordings: list of int
        Number of recordings to load.

    Returns
    -------
    windows_ds: BaseConcatDataset
        Windowed data.

    .. warning::
        The recordings from the TUH Abnormal corpus do not all share the same
        sampling rate. The following assumes that the files have already been
        resampled to a common sampling rate.

    """
    if recording_ids is None:
        recording_ids = list(range(n_recordings))

    ds = TUHAbnormal(
        TUH_PATH,
        recording_ids=recording_ids,
        target_name="pathological",
        preload=preload,
    )

    fs = ds.datasets[0].raw.info["sfreq"]
    window_len_samples = int(fs * window_len_s)
    window_stride_samples = int(fs * 4)
    # window_stride_samples = int(fs * window_len_s)
    windows_ds = create_fixed_length_windows(
        ds,
        start_offset_samples=10 * fs,  # Skip first 10 seconds to avoid artifacts.
        stop_offset_samples=None,
        window_size_samples=window_len_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=True,
        preload=preload,
        n_jobs=8
    )

    # Drop bad epochs
    # XXX: This could be parallelized.
    # XXX: Also, this could be implemented in the Dataset object itself.
    # We don't support drop_bad since the last version braindecode,
    # to optimize the dataset speed. If you know how to fix, please open a PR.
    # for ds in windows_ds.datasets:
    #    ds.raw.drop_bad()
    #   assert ds.raw.preload == preload

    return windows_ds


def plot_window(window_data, ch_names, sfreq, out_path, pathological_label=None):
    """Save one EEG window as an offset trace plot."""
    window_uv = window_data * 1e6
    n_channels, n_times = window_uv.shape
    time_s = np.arange(n_times) / sfreq

    # Use a robust channel spacing to avoid overlap while keeping dynamic range.
    spacing = np.percentile(np.abs(window_uv), 95) * 2.2
    spacing = max(spacing, 20.0)

    fig, ax = plt.subplots(figsize=(14, 9))
    offsets = np.arange(n_channels)[::-1] * spacing

    for ch_idx in range(n_channels):
        ax.plot(
            time_s,
            window_uv[ch_idx] + offsets[ch_idx],
            linewidth=0.8,
            color="black",
            alpha=0.75,
        )

    if ch_names is None:
        plot_ch_names = [f"CH{ch_idx:02d}" for ch_idx in range(n_channels)]
    else:
        plot_ch_names = list(ch_names)
        if len(plot_ch_names) < n_channels:
            start = len(plot_ch_names)
            plot_ch_names.extend([f"CH{ch_idx:02d}" for ch_idx in range(start, n_channels)])
        elif len(plot_ch_names) > n_channels:
            plot_ch_names = plot_ch_names[:n_channels]

    ax.set_yticks(offsets)
    ax.set_yticklabels(plot_ch_names)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (offset traces)")
    title = "Single EEG window"
    if pathological_label is not None:
        title += f" | pathological={pathological_label}"
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if pathological_label is not None:
        ax.text(
            0.01,
            0.99,
            f"pathological: {pathological_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _get_recording_pathological_label(recording_ds):
    """Read recording-level pathological label from a window dataset."""
    if hasattr(recording_ds, "y") and recording_ds.y is not None:
        y = np.asarray(recording_ds.y).reshape(-1)
        if y.size > 0:
            return bool(y[0])

    description = getattr(recording_ds, "description", None)
    if description is not None and hasattr(description, "get"):
        value = description.get("pathological", None)
        if value is not None:
            return bool(value)

    return None


def _pick_window_index_in_recording(
    start_idx,
    n_windows,
    start_ratio=0.2,
    end_ratio=0.8,
    target_ratio=0.5,
):
    """Pick one window index from the interior of a recording."""
    if n_windows <= 0:
        return None

    max_local_idx = n_windows - 1
    start_local = int(np.floor(max_local_idx * start_ratio))
    end_local = int(np.ceil(max_local_idx * end_ratio))

    start_local = int(np.clip(start_local, 0, max_local_idx))
    end_local = int(np.clip(end_local, 0, max_local_idx))
    if end_local < start_local:
        start_local, end_local = end_local, start_local

    target_local = int(round(max_local_idx * target_ratio))
    target_local = int(np.clip(target_local, start_local, end_local))
    return start_idx + target_local


def collect_window_indices_by_label(
    windows_ds,
    start_ratio=0.2,
    end_ratio=0.8,
    target_ratio=0.5,
):
    """Return representative global window indices split by pathological label."""
    pathological_indices, healthy_indices = [], []
    start_idx = 0

    for recording_ds in windows_ds.datasets:
        n_windows = len(recording_ds)
        label = _get_recording_pathological_label(recording_ds)

        # Fallback in case dataset internals differ.
        if label is None and n_windows > 0:
            label = bool(windows_ds[start_idx][1])

        chosen_idx = _pick_window_index_in_recording(
            start_idx=start_idx,
            n_windows=n_windows,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            target_ratio=target_ratio,
        )

        target = pathological_indices if label else healthy_indices
        if chosen_idx is not None:
            target.append(chosen_idx)
        start_idx += n_windows

    return pathological_indices, healthy_indices


def choose_evenly_spaced(indices, n_examples):
    """Select examples across the full class span instead of only first windows."""
    if len(indices) <= n_examples:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=n_examples, dtype=int)
    return [indices[p] for p in positions]


def choose_balanced_recording_ids(n_per_class):
    """Pick recording IDs from both pathological classes."""
    ds_all = TUHAbnormal(
        TUH_PATH,
        target_name="pathological",
        preload=False,
    )
    description = ds_all.description.reset_index(drop=True)

    pathological_mask = description["pathological"].astype(bool)
    pathological_ids = description.index[pathological_mask].tolist()
    healthy_ids = description.index[~pathological_mask].tolist()

    selected_pathological = choose_evenly_spaced(pathological_ids, n_per_class)
    selected_healthy = choose_evenly_spaced(healthy_ids, n_per_class)
    return selected_pathological + selected_healthy


def get_recording_dataset_for_window(windows_ds, window_idx):
    """Map global window index to its source recording window dataset."""
    start_idx = 0
    for recording_ds in windows_ds.datasets:
        stop_idx = start_idx + len(recording_ds)
        if start_idx <= window_idx < stop_idx:
            return recording_ds
        start_idx = stop_idx
    return windows_ds.datasets[0]


print("Loading data...")
recording_ids = choose_balanced_recording_ids(n_per_class=25)
print("Selected recording IDs:", len(recording_ids), "(balanced pathological/healthy)")
windows_ds = load_example_data(
    preload=True,
    window_len_s=35,
    n_recordings=len(recording_ids),
    recording_ids=recording_ids,
)
print("Data loaded.")

print("data shape : ", windows_ds)

feature_cols = [
    "year",
    "month",
    "day",
    "path",
    "subject",
    "session",
    "segment",
    "age",
    "gender",
    "version",
    "train",
    "pathological",
]

available_cols = [c for c in feature_cols if c in windows_ds.description.columns]
print(f"\nRecording-level features {len(available_cols)} columns):")
print(windows_ds.description[available_cols].to_string(index=False))


sampling_start_ratio = 0.20
sampling_end_ratio = 0.80
sampling_target_ratio = 0.50

pathological_indices, healthy_indices = collect_window_indices_by_label(
    windows_ds,
    start_ratio=sampling_start_ratio,
    end_ratio=sampling_end_ratio,
    target_ratio=sampling_target_ratio,
)
print("\nSampling configuration:")
print("start_ratio:", sampling_start_ratio)
print("end_ratio:", sampling_end_ratio)
print("target_ratio:", sampling_target_ratio)

print("\nCandidate windows by label (one per recording):")
print("pathological=True candidates:", len(pathological_indices))
print("pathological=False candidates:", len(healthy_indices))

if len(pathological_indices) == 0 or len(healthy_indices) == 0:
    print("Warning: one class has zero windows. Increase recordings or rebalance selection.")

n_examples_per_class = 4
selected_pathological = choose_evenly_spaced(pathological_indices, n_examples_per_class)
selected_healthy = choose_evenly_spaced(healthy_indices, n_examples_per_class)

print("\nSelected pathological indices:", selected_pathological)
print("Selected healthy indices:", selected_healthy)

for class_name, chosen_indices in [
    ("pathological", selected_pathological),
    ("healthy", selected_healthy),
]:
    for rank, window_idx in enumerate(chosen_indices):
        sample = windows_ds[window_idx]
        window_x = np.asarray(sample[0])
        pathological_label = bool(sample[1])
        recording_ds = get_recording_dataset_for_window(windows_ds, window_idx)
        fs = recording_ds.raw.info["sfreq"]
        ch_names = recording_ds.raw.ch_names
        plot_path = f"window_{class_name}_{rank:02d}_idx{window_idx:05d}.png"
        plot_window(
            window_x,
            ch_names,
            fs,
            plot_path,
            pathological_label=pathological_label,
        )
        print(
            f"saved {plot_path} | window_idx={window_idx} | pathological={pathological_label}"
        )
