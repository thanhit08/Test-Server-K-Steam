# import tensorflow as tf
import torch


def manhattan_similarity(x1, x2):
    """
    Similarity function based on manhattan distance and exponential function.
    Args:
        x1: x1 input vector
        x2: x2 input vector

    Returns: Similarity measure in range between 0...1,
    where 1 means full similarity and 0 means no similarity at all.

    """
    manhattan_sim = torch.exp(-manhattan_distance(x1, x2))
    return manhattan_sim


def manhattan_distance(x1, x2):
    """
    Also known as l1 norm.
    Equation: sum(|x1 - x2|)
    Example:
        x1 = [1,2,3]
        x2 = [3,2,1]
        MD = (|1 - 3|) + (|2 - 2|) + (|3 - 1|) = 4
    Args:
        x1: x1 input vector
        x2: x2 input vector

    Returns: Manhattan distance between x1 and x2. Value grater than or equal to 0.

    """
    length = x1.size(1)
    # print(f"vector length: {length}, {x1.shape}, {x2.shape}")
    # result =1. - (torch.sum(torch.abs(x1 - x2), axis=1, keepdims=True))
    result = 1. - (torch.sum((x1 - x2), axis=1, keepdims=True))
    return result

    return 1. - (torch.sum(torch.abs(x1 - x2), axis=1, keepdims=True))


def manhattan_distance_w_max(x1, x2, max_value):
    """
    Also known as l1 norm.
    Equation: sum(|x1 - x2|)
    Example:
        x1 = [1,2,3]
        x2 = [3,2,1]
        MD = (|1 - 3|) + (|2 - 2|) + (|3 - 1|) = 4
    Args:
        x1: x1 input vector
        x2: x2 input vector

    Returns: Manhattan distance between x1 and x2. Value grater than or equal to 0.

    """
    length = x1.size(1)
    # print(f"manhattan_distance_w_max : {x1.size()}")
    # print(f"vector length: {length}, {x1.shape}, {x2.shape}")
    # result = max_value - (torch.sum(torch.abs(x1 - x2), axis=1, keepdims=True))
    result = max_value - (torch.sum(torch.abs(x1 - x2), axis=1))
    return result


def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum(torch.square(x1 - x2), axis=1, keepdims=True))


def cosine_distance(x1, x2):
    # TODO consider adding for case when input vector contains only 0 values, eps = 1e-08
    num = torch.sum(x1 * x2, axis=1)
    denom = torch.sqrt(torch.sum(torch.square(x1), axis=1)) * \
        torch.sqrt(torch.sum(torch.square(x2), axis=1))
    cos_sim = torch.unsqueeze(torch.div(num, denom), -1)
    return cos_sim
