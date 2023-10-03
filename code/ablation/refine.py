import json
from PACKAGE import asyncThread
import os
import random
import threading
import queue


def load_api_key(index):
    with open('../../keys.json', 'r') as f:
        api_keys = json.load(f)
    return api_keys[index]


api_key = ''


class Summarize:

    @staticmethod
    def traverse_path(folder_path):
        for root, dirs, files in os.walk(folder_path):
            if files and dirs:
                Summarize.traverse_sub_path(root)

    @staticmethod
    def traverse_sub_path(path):
        if not os.path.exists(os.path.join(path, 'refine')):
            os.makedirs(os.path.join(path, 'refine'))
        queries, articles = SelectSummary.load_query_article(path)

        queries, articles = SelectSummary.load_query_article(path)
        t_list = []
        results = queue.Queue()

        def worker(index, article):
            result = Summarize.refine_summary(queries[index], article)
            results.put((index, result))

        for index, article in enumerate(articles):
            t = threading.Thread(target=worker, args=(index, article))
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()

        summary_results = [result for index, result in sorted(results.queue)]
        with open(os.path.join(path, 'refine/gpt3_all_summary.json'), 'w') as f:
            temp = json.dumps(summary_results, indent=4)
            f.write(temp)

        summary_results = [item[len(item) - 1] for item in summary_results]
        with open(os.path.join(path, 'refine/gpt3_summary.json'), 'w') as f:
            temp = json.dumps(summary_results, indent=4)
            f.write(temp)

    @staticmethod
    def refine_summary(query, docs):
        # l2h
        # docs.reverse()
        intermediate_outputs = ['']
        # chain refine
        for index, doc in enumerate(docs):
            map_prompt = f"Write an answer based on the following question, the given meeting and the based information. Try to answer thoroughly and do not leave out useful information.\n MEETING:{doc}\n QUESTION:{query}\nBASED INFORMATION{intermediate_outputs[index]}\n SUMMARY: \n"
            system = "You are a helpful assistant that gives long answer to question based on a long meeting."
            messages = [[{"role": "system", "content": system},
                         {"role": "user", "content": map_prompt}]]
            intermediate_output = asyncThread.run(messages=messages,
                                                  engine_name="gpt-3.5-turbo-16k-0613",
                                                  # engine_name="gpt-4-0613",
                                                  temperature=0.7,
                                                  max_tokens=600,
                                                  top_p=0.9,
                                                  api_key=api_key,
                                                  requests_per_minute=20)
            intermediate_outputs.append(intermediate_output[0])
        return intermediate_outputs


class SelectSummary:
    @staticmethod
    def random_select(count):
        indexes = random.sample(range(260), count)
        with open('QMSum/randomIndex/index.json', 'w') as f:
            temp = json.dumps(indexes)
            f.write(temp)

    @staticmethod
    def load_query_article(path):
        filename = 'test.json'
        with open('QMSum/randomIndex/index.json', 'r') as f:
            index_list = json.load(f)
        with open(os.path.join(path, filename), 'r') as f:
            data = json.load(f)
            queries = [item['Query'] for item in data]
        with open(os.path.join(path, 'split/split_' + filename), 'r') as f:
            articles = json.load(f)
        queries = [queries[index] for index in index_list]
        articles = [articles[index] for index in index_list]
        return queries, articles


Summarize.traverse_sub_path('QMSum/sparse/MIN')
# Summarize.traverse_sub_path('QMSum/oracle')
# path = 'QMSum/oracle'
# with open(os.path.join(path, 'refine/gpt3_all_summary.json'), 'r') as f:
#     summary_results = json.load(f)
#
# summary_results = [item[1] for item in summary_results]
# with open(os.path.join(path, 'refine/gpt3_summary.json'), 'w') as f:
#     temp = json.dumps(summary_results, indent=4)
#     f.write(temp)
