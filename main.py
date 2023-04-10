from src.nearest_neighbors.faiss import FaissNearestNeighbors
from src.nearest_neighbors.hnswlib import HNSWNearestNeighbors

if __name__ == '__main__':
    _embeddings = [
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
    ]

    query = [1.1, 2.3, 3.2]
    k = 3

    f = FaissNearestNeighbors(_embeddings)
    # Perform nearest neighbor search
    result = f.search(query, k)

    print(f"result: {result}")

    h = HNSWNearestNeighbors(_embeddings)
    result2 = h.search(query, k)
    print("result2: ", result2)
