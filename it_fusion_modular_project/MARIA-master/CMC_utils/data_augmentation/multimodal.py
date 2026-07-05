import numpy as np

__all__ = ["multimodal_missing_augmentation"]


def multimodal_missing_augmentation(sample: list) -> list:
    """
    Randomly mask a fraction of the sample's modalities
    Parameters
    ----------
    sample : list

    Returns
    -------
    list
    """
    not_missing_idx = [i for i in range(len(sample)) if not np.isnan(sample[i]).all()]
    to_mask = np.random.choice([False, True], p=[0.5, 0.5])
    if to_mask and len(not_missing_idx) > 2:
        idx_to_mask = np.random.choice(not_missing_idx, size=np.random.choice(np.arange(1, len(not_missing_idx) - 1)), replace=False)
        for i in idx_to_mask:
            sample[i] = np.full(sample[i].shape, np.nan)
    return sample


if __name__ == "__main__":
    pass
