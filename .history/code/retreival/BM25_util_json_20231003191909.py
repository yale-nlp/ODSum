import sys 
sys.path.append("./util")
from mds import *

SEPARATOR = " <doc-sep> "

def from_divided_document_to_primera(documents):
    result = ""
    for document in documents:
        
        IS_LAST_DOCUMENT = document == documents[-1]

        # delete every \n
        document = document.replace("\n", "")
        
        # delete \"
        document = document.replace("\"", "")
        
        result += document

        # add the separator between the documents
        result += SEPARATOR if (len(documents) > 1 and IS_LAST_DOCUMENT==False) else ""
        
    result += ""
    return result




import json

def output_to_file(set_name, retrieved_number, retrieved_method):

    if set_name == 'train':
        queries_set =  queries[0:50]
    elif set_name == 'test':
        queries_set =  queries[50:102]
    elif set_name == 'dev':
        queries_set =  queries[102:]


    documents = ""

    data = []

    for one_doc_query in queries_set:

        for query in one_doc_query:

            # get the top 3 documents
            retrieved_docs = retrieve(query['question_text'], retrieved_number)
            documents = []
            for doc in retrieved_docs:
                documents.append(doc[1])
            
            article = from_divided_document_to_primera(documents)

            summary_1 = query['responses'][0]['response_text']
            summary_2 = query['responses'][1]['response_text']
            summary_3 = query['responses'][2]['response_text']
            summary_4 = query['responses'][3]['response_text']

            data.append({
                'Query': query['question_text'],
                'Summary_1': summary_1,
                'Summary_2': summary_2,
                'Summary_3': summary_3,
                'Summary_4': summary_4,
                'Article': article
            })

    # Write the list of dictionaries to a JSON file
    with open(f'./{retrieved_method}/{set_name}.json', 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)


