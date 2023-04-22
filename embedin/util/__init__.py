import numpy as np


def to_np_array(embeddings):
    """
    Converts a list of embeddings to a numpy array.

    Args:
        embeddings: A list of embeddings, where each element is either a scalar or a list of scalars.

    Returns:
        A numpy array representing the input embeddings.
    """
    # Convert the input list to a numpy array
    embeddings_array = np.array(embeddings)

    # If the array has only one dimension, add an extra dimension to represent a single example
    if len(embeddings_array.shape) == 1:
        embeddings_array = embeddings_array[np.newaxis, :]

    return embeddings_array
