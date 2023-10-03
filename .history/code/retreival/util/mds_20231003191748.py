from datasets import load_dataset
# dataset = load_dataset("json", data_files={"train": "../../data/divided/train.jsonl", "test": "../../data/divided/test.jsonl", "dev": "../../data/divided/dev.jsonl"})

# retrieve the data using BM25
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

print(1)

stop_words_extended = stop_words.copy()
# add stop words to stop_words_extended
stop_words_extended.add('plot')
stop_words_extended.add('story')
stop_words_extended.add('setting')

# extract the data from the dataset
# document is a list of tuples, in the form of (passage_id, divided_document)

documents = []
for i in range(len(dataset['train'])):
    for document in dataset['train'][i]['divided_document']:
        documents.append((dataset['train'][i]['metadata']['passage_id'], document))
for i in range(len(dataset['test'])):
    for document in dataset['test'][i]['divided_document']:
        documents.append((dataset['test'][i]['metadata']['passage_id'], document))
for i in range(len(dataset['dev'])):
    for document in dataset['dev'][i]['divided_document']:
        documents.append((dataset['dev'][i]['metadata']['passage_id'], document))

documents = list(set(documents))

# average number of documents per passage
average_doc_num = 0
for i in range(len(dataset['train'])):
    average_doc_num += len(dataset['train'][i]['divided_document'])
for i in range(len(dataset['test'])):
    average_doc_num += len(dataset['test'][i]['divided_document'])
for i in range(len(dataset['dev'])):
    average_doc_num += len(dataset['dev'][i]['divided_document'])
average_doc_num /= len(dataset['train']) + len(dataset['test']) + len(dataset['dev'])
print(average_doc_num)
# find how many passages have only one documents
one_doc_num = 0
for i in range(len(dataset['train'])):
    if len(dataset['train'][i]['divided_document']) == 1:
        one_doc_num += 1
for i in range(len(dataset['test'])):
    if len(dataset['test'][i]['divided_document']) == 1:
        one_doc_num += 1
for i in range(len(dataset['dev'])):
    if len(dataset['dev'][i]['divided_document']) == 1:
        one_doc_num += 1
print("in total there are" , one_doc_num  ,"single document passages")

def preprocess(query, stop_words_pre = stop_words_extended):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    query = word_tokenize(query)
    query = [word for word in query if not word in stop_words_pre]
    query = [lemmatizer.lemmatize(word) for word in query]
    query = [ps.stem(word) for word in query]
    return query

tokenized_corpus = [preprocess(document[1]) for document in documents]



# read form data/modified_queries/query.jsonl
with open('../../data/modified_queries/query.jsonl') as f:
    queries = f.readlines()
queries = [json.loads(query) for query in queries]
print("the number of the queries" , len(queries))



bm25 = BM25Okapi(tokenized_corpus)

# given a query, return the top k documents
def retrieve(query, k , retriever = bm25):
    if(retriever == bm25):
        tokenized_query = preprocess(query)
        # print(tokenized_query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_k = np.argsort(doc_scores)[::-1][:k]
        return [documents[i] for i in top_k]
    

# measure the percentage of query that had the correct answer
def precision(query, id , k):
    correct = 0
    top_k = retrieve(query, k)
    for doc in top_k:
        if doc[0] == id:
            correct += 1
    return correct / k

def recall(query, id, k):
    correct_retrieved = 0
    top_k = retrieve(query, k)
    for doc in top_k:
        if doc[0] == id:
            correct_retrieved += 1
    all_relevant = len([document for document in documents if document[0] == id])
    return correct_retrieved / all_relevant

# iterate through every question and calculate the accuracy
def precision_all(k, prefix = '', delete_stop_words = False , modified = False ,queries = queries):
    total = 0
    correct = 0


    # modified queries
    if modified:
        for i in range(len(dataset['train'])):
            title = dataset['train'][i]['title']
            id = dataset['train'][i]['metadata']['passage_id']
            for query in queries:
                # find the query that has the same title as the passage
                if query[0]['title'] == title:
                    for question in query:
                        question_text = question['question_text']
                        total += 1
                        correct += precision(question_text, id, k)
                    break

        for i in range(len(dataset['test'])):

            title = dataset['test'][i]['title']
            id = dataset['test'][i]['metadata']['passage_id']
            for query in queries:
                # find the query that has the same title as the passage
                if query[0]['title'] == title:
                    for question in query:
                        question_text = question['question_text']
                        total += 1
                        correct += precision(question_text, id, k)
                    break
            
        for i in range(len(dataset['dev'])):
            title = dataset['dev'][i]['title']
            id = dataset['dev'][i]['metadata']['passage_id']
            for query in queries:
                # find the query that has the same title as the passage
                if query[0]['title'] == title:
                    for question in query:
                        question_text = question['question_text']
                        total += 1
                        correct += precision(question_text, id, k)
                    break
        return correct / total
    


    # not modified queries
    else :
        for i in range(len(dataset['train'])):
            id = dataset['train'][i]['metadata']['passage_id']
            for question in dataset['train'][i]['questions']:
                query = question['question_text']
                if delete_stop_words:      
                    if  ( len(preprocess(query)) < 2):
                        continue
                total += 1
                correct += precision(query, id, k)
        for i in range(len(dataset['test'])):    
            id = dataset['test'][i]['metadata']['passage_id']
            for question in dataset['test'][i]['questions']:
                query = question['question_text']
                if delete_stop_words:    
                    if  ( len(preprocess(query)) < 2):
                        continue
                total += 1
                correct += precision(query, id, k)   
        for i in range(len(dataset['dev'])):
            id = dataset['dev'][i]['metadata']['passage_id']
            for question in dataset['dev'][i]['questions']:
                query = question['question_text']
                if delete_stop_words:  
                    if  ( len(preprocess(query)) < 2):
                        continue
                total += 1
                correct += precision(query, id, k)
        return correct / total
    


def recall_all(k, prefix = '' , delete_stop_words = False , modified = False , queries = queries): 
    total = 0
    correct = 0


    # modified queries

    if modified:
        for i in range(len(dataset['train'])):
            title = dataset['train'][i]['title']
            id = dataset['train'][i]['metadata']['passage_id']
            for query in queries:
                # find the query that has the same title as the passage
                if query[0]['title'] == title:
                    for question in query:
                        question_text = question['question_text']
                        total += 1
                        correct += recall(question_text, id, k)
                    break

        for i in range(len(dataset['test'])):
                
                title = dataset['test'][i]['title']
                id = dataset['test'][i]['metadata']['passage_id']
                for query in queries:
                    # find the query that has the same title as the passage
                    if query[0]['title'] == title:
                        for question in query:
                            question_text = question['question_text']
                            total += 1
                            correct += recall(question_text, id, k)
                        break

        for i in range(len(dataset['dev'])):
            title = dataset['dev'][i]['title']
            id = dataset['dev'][i]['metadata']['passage_id']
            for query in queries:
                # find the query that has the same title as the passage
                if query[0]['title'] == title:
                    for question in query:
                        question_text = question['question_text']
                        total += 1
                        correct += recall(question_text, id, k)
                    break
        return correct / total


    # not modified queries
    else:
        for i in range(len(dataset['train'])):
            id = dataset['train'][i]['metadata']['passage_id']
            for question in dataset['train'][i]['questions']:
                query = question['question_text']
                if delete_stop_words:
                    if  ( len(preprocess(query)) < 2):
                        continue
                total += 1
                correct += recall(query, id, k)
        for i in range(len(dataset['test'])):    
            id = dataset['test'][i]['metadata']['passage_id']
            for question in dataset['test'][i]['questions']:
                query = question['question_text']
                if delete_stop_words:
                    if  ( len(preprocess(query)) < 2):
                        continue
                total += 1
                correct += recall(query, id, k)   
        for i in range(len(dataset['dev'])):
            id = dataset['dev'][i]['metadata']['passage_id']
            for question in dataset['dev'][i]['questions']:
                query = question['question_text']
                if delete_stop_words:
                    if  ( len(preprocess(query)) < 2):
                        continue
                total += 1
                correct += recall(query, id, k)
        return correct / total

def f1(k, prefix = '' , delete_stop_words = False, modified = False ,queries = queries):
    p = precision_all(k, prefix , delete_stop_words , modified , queries = queries)
    r = recall_all(k, prefix , delete_stop_words , modified , queries = queries)
    return 2 * p * r / (p + r)