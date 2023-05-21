# Python API Reference

## 1. Constructor

### Client(collection_name, url=None, embedding_fn="sentence_transformer", index_hint=None, debug=False, **kwargs)

Initializes a new client object.

__Parameters__

| Param              | Type                          | Description                                                                 |
|:-------------------|:------------------------------|:----------------------------------------------------------------------------|
| `collection_name`  | _str_                         | Name of the collection.                                                     |
| `url`              | _str, optional_               | Database URL. Defaults to None.                                              |
| `embedding_fn`     | _str or EmbeddingFunction_    | The embedding function to use. The default is the SentenceTransformer model. |
| `index_hint`       | _str, optional_               | Similarity search index to use. Supports: 'flat' and 'hnsw'.                |
| `debug`            | _bool, optional_              | Enable debug mode. Defaults to False.                                        |
| `**kwargs`         |                               | Additional keyword arguments.                                               |

__Attributes__

| Attribute              | Type                         | Description                                                                 |
|:-----------------------|:-----------------------------|:----------------------------------------------------------------------------|
| `collection_id`        | _int_                        | ID of the collection.                                                        |
| `embedding_fn`         | _str or EmbeddingFunction_    | Embedding function to use. Supported options are 'sentence_transformer' and 'openai'. |
| `session`              | _Session_                    | SQLAlchemy session.                                                          |
| `collection_service`   | _CollectionService_          | CollectionService instance.                                                  |
| `embedding_service`    | _EmbeddingService_           | EmbeddingService instance.                                                   |
| `embedding_rows`       | _List[EmbeddingModel]_       | List of Embedding instances for the current collection.                      |

__Methods__

| Method                  | Description                                                                 |
|:------------------------|:----------------------------------------------------------------------------|
| `create_or_get_collection(name)` | Get the ID of an existing collection or create a new one.              |
| `create_collection(name)`        | Create a new collection with the given name.                           |
| `get_collection(name)`           | Get the ID of an existing collection with the given name.              |
| `add_data(texts, meta_data=None)` | Add new data to the collection.                                        |
| `query(query_texts, top_k=3)`     | Find nearest neighbors for the given query text(s).                     |

__Example__

```py
from embedin import Embedin

url = 'sqlite:///test.db'
client = Embedin(collection_name="test_collection", texts=["This is a test", "Hello world!"], url=url)

# add more texts
client.add_data(["It's a good day!", "Don't work too hard."], meta_data=[{"source": "abc"}, {"source": "efg"}])

result = client.query("These are tests", top_k=2)
```

## 2. add_data(texts, meta_data=None)
WIP

## 3. create_collection(name)
WIP

## 4. get_collection(name)
WIP

## 5. query(query_texts, top_k=3)
WIP
