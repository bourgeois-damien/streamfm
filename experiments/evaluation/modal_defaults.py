from __future__ import annotations

MODAL_VOLUME_ROOT = "/data"


def modal_default_data_path(
    task: str,
    split: str,
    volume_root: str = MODAL_VOLUME_ROOT,
) -> str:
    """Return the Modal volume dataset path expected by the evaluation configs."""
    normalized_task = task.lower().replace("-", "_")
    normalized_split = split.lower().strip()
    dataset_root = f"{volume_root}/datasets"
    if normalized_task in {"se", "melflow", "stftpr"}:
        return f"{dataset_root}/EARS-WHAM_v2_16k/{normalized_split}.csv"
    if normalized_task == "derev":
        return f"{dataset_root}/EARS-Reverb_v2_16k/{normalized_split}.csv"
    if normalized_task == "bwe":
        return f"{dataset_root}/EARS_v2_16k_BWR/{normalized_split}"
    if normalized_task == "lyra":
        return f"{dataset_root}/EARS_v2_16k_Lyra/dataset_highpass75/3200bit/{normalized_split}"
    raise ValueError(f"Unsupported task '{task}'.")


def resolve_modal_data_path(
    data_path: str,
    *,
    task: str,
    split: str,
    volume_root: str = MODAL_VOLUME_ROOT,
) -> str:
    """Use an explicit data path when given, otherwise choose the Modal default."""
    if data_path:
        return data_path
    return modal_default_data_path(task, split, volume_root=volume_root)
