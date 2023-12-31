README for Retrieval Code
======================================

Description:
------------

This module provides functionalities for extracting and processing embeddings from text data using multiple retrieval methods such as Contriever, BM25, and a generic LLM embedding. The purpose is to help with the information retrieval from a collection of meeting texts based on given queries.

Dependencies:
-------------

- json
- os
- numpy
- math
- torch
- tqdm
- sklearn
- openai (for LLM embeddings)

Usage:
------

1. **Embedding Generation**

   Use the `get_embedding` function to generate embeddings for a given piece of text.

2. **Information Retrieval Class**

   The `InformationRetrieval` class is used for loading, processing, and querying the meeting texts based on various algorithms.

   **Methods include**:
   
   - contriever_embedding: To retrieve meetings using Contriever.
   - precompute_embeddings: To precompute embeddings for faster querying.
   - precompute_contriever_embeddings: To precompute Contriever embeddings.
   - llm_embedding: To retrieve meetings using LLM embeddings.
   - bm25: To retrieve meetings using BM25.
   - evaluate_performance: To evaluate the retrieval's performance.
   - save_splits: To save results in train, test, and dev splits.
   - run_evaluation: To evaluate the methods for given top-k values.

Configuration:
--------------

The module supports configuration via the following global variables:

- `USE_CONTRIEVER`: If set to True, Contriever will be used for embeddings.
- `USE_BM25`: If set to True, BM25 will be used for retrieval.
- `USE_LLM_EMBEDDING`: If set to True, the LLM embeddings will be used.

**Note**:
- Remember to set the OpenAI API key if using LLM embeddings.
- Ensure that the Contriever model and tokenizer are available if using Contriever.
  

Usage Example:
--------------

```python
# Initialize the Information Retrieval system with your data
ir_system = InformationRetrieval(query_file="path/to/queries.json", meeting_folder="path/to/meetings")

# Use BM25 to get top 3 meetings for a given query
top_meetings = ir_system.bm25("Your sample query", 3)

print(top_meetings)
```

Note:
-----

Always be cautious about sharing and using API keys in scripts. Do not push them to public repositories.


Contribution:
-------------

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.