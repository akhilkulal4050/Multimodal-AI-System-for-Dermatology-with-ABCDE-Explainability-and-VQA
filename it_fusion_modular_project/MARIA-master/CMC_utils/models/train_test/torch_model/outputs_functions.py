import torch

__all__ = ["surpass_threshold", "max_index"]


def max_index(outputs, **_) -> torch.Tensor:
    """
    Returns the index of the maximum value in the output tensor.
    Parameters
    ----------
    outputs : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    _, prediction = torch.max(outputs, 1)
    return prediction


def surpass_threshold(outputs, threshold=0.5, **_) -> torch.Tensor:
    """
    Returns the prediction for the binary classification model. The prediction is 1 if the output surpasses the threshold
    and 0 otherwise.
    Parameters
    ----------
    outputs : torch.Tensor
    threshold : float

    Returns
    -------
    torch.Tensor
    """
    preds = (outputs > threshold).long()
    return preds


if __name__ == "__main__":
    pass
