import json

import sys
sys.path.append('./contriever')  # Add the directory containing the Contriever module to the Python path

# print(sys.path)


# Define the path to the JSON file
file_path = '../LLM-embedding/doc_embeddings.json'

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# read form data/modified_queries/query.jsonl
with open('../../data/modified_queries/query_title_answer.jsonl') as f:
    queries = f.readlines()
queries = [json.loads(query) for query in queries]

# Function to get ground truth for each query
def get_ground_truth(query_title, data, passage_id_to_title):
    ground_truth_set = set()
    for i, doc in enumerate(data):
        if passage_id_to_title.get(doc['passage_id']) == query_title:
            ground_truth_set.add(i)
    return ground_truth_set


# Map from passage_id to title (this is inside passage_id_to_title.json)
with open('./passage_id_to_title.json') as f:
    passage_id_to_title = json.load(f)






# Perform retrieval and evaluation

min_num = 3

import numpy as np
from tqdm import tqdm


from src.contriever import Contriever
from transformers import AutoTokenizer

# Initialize Contriever and tokenizer
contriever = Contriever.from_pretrained("facebook/contriever")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

# Initialize list for Contriever-based evaluation metrics
precision_list_contr = []
recall_list_contr = []
f1_list_contr = []

# Tokenize data and obtain embeddings
inputs_data = tokenizer([doc['text'] for doc in data], padding=True, truncation=True, return_tensors="pt")
data_embeddings = contriever(**inputs_data)

# Perform retrieval and evaluation using Contriever
for query_set in queries:
    for query_dict in query_set:
        query_text = query_dict['question_text']
        
        # Tokenize and get embeddings for the query
        inputs_query = tokenizer([query_text], padding=True, truncation=True, return_tensors="pt")
        query_embedding = contriever(**inputs_query)
        
        # Compute similarity scores and retrieve top documents
        dot_products = (query_embedding @ data_embeddings.T).squeeze()
        top_docs = dot_products.argsort(descending=True)[:3]  # Top 3 documents
        

        # Get ground truth set for this query
        ground_truth_set = get_ground_truth(query_title, data, passage_id_to_title)
        
        # Evaluate
        retrieved_set = set(top_docs)
        true_positive = len(retrieved_set.intersection(ground_truth_set))
        precision = true_positive / len(retrieved_set)
        recall = true_positive / len(ground_truth_set)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Append to lists
        precision_list_emb.append(precision)
        recall_list_emb.append(recall)
        f1_list_emb.append(f1)


        # Save top-retrieved documents
        top_docs_text = "<doc-sep>".join([data[i]['text'] for i in top_docs])
        result = {
            "Query": query_text,
            "Article": top_docs_text,
            "Summary_1": query_dict['responses'][0]['response_text'],
            "Summary_2": query_dict['responses'][1]['response_text'],
            "Summary_3": query_dict['responses'][2]['response_text'],
            "Summary_4": query_dict['responses'][3]['response_text']
        }
        results.append(result)

# Save results to JSON file
with open("min_retrieved_documents.json", "w") as f:
    json.dump(results, f, indent=4)

# Calculate average metrics
avg_precision_emb = sum(precision_list_emb) / len(precision_list_emb)
avg_recall_emb = sum(recall_list_emb) / len(recall_list_emb)
avg_f1_emb = sum(f1_list_emb) / len(f1_list_emb)

print(f"Embedding-Based Retrieval - Average Precision: {avg_precision_emb}")
print(f"Embedding-Based Retrieval - Average Recall: {avg_recall_emb}")
print(f"Embedding-Based Retrieval - Average F1-score: {avg_f1_emb}")