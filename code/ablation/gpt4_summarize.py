import json
from PACKAGE import asyncThread
import os
import random

api_key = ''


class SelectSummary:
    @staticmethod
    def random_select(count):
        indexes = random.sample(range(131), count)
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
        with open(os.path.join(path, 'split/split8000_' + filename), 'r') as f:
            articles = json.load(f)
        queries = [queries[index] for index in index_list]
        articles = [articles[index] for index in index_list]
        return queries, articles


class Summarize:

    @staticmethod
    def traverse_path(folder_path):
        for root, dirs, files in os.walk(folder_path):
            if files and dirs:
                Summarize.traverse_sub_path(root)

    @staticmethod
    def traverse_sub_path(path):
        queries, articles = SelectSummary.load_query_article(path)

        save_intermediate_outputs = []

        intermediate_outputs = Summarize.intermediate_summary(queries, articles)

        save_intermediate_outputs.append(intermediate_outputs)
        save_intermediate_outputs = save_intermediate_outputs[0]

        with open(os.path.join(path, 'summary/gpt4_summary.json'), 'w') as f:
            temp = json.dumps(save_intermediate_outputs, indent=4)
            f.write(temp)

    @staticmethod
    def intermediate_summary(querys, docs):
        # map_prompts = [
        #     f"Write an answer based on the following question and the given meeting.Try to answer thoroughly and do not leave out useful information.\n MEETING:{doc[0]}\n QUESTION:{query}\n SUMMARY: \n"
        #     for doc, query in zip(docs, querys)
        # ]
        # map_prompts = [
        #     f"Abstract the paragraph from the meeting which can be used to answer the question. Do not leave out useful information.\n MEETING:{doc}\n QUESTION:{query}\n ABSTRACTED PARAGRAPH: \n"
        #     for doc in docs]
        map_prompts = [
            f"Write an answer based on the following question and the given story.Try to answer thoroughly and do not leave out useful information.\n STORY:{doc[0]}\n QUESTION:{query}\n SUMMARY: \n"
            for doc, query in zip(docs, querys)
        ]
        # system = "You are a helpful assistant that gives long answer to question based on a long meeting."
        system = "You are a helpful assistant that gives long answer to question based on a long story."
        messages = [[{"role": "system", "content": system},
                     {"role": "user", "content": map_prompt}] for map_prompt in map_prompts]
        intermediate_outputs = asyncThread.run(messages=messages,
                                               # engine_name="gpt-3.5-turbo-16k-0613",
                                               engine_name="gpt-4-0613",
                                               temperature=0.7,
                                               max_tokens=600,
                                               top_p=0.9,
                                               api_key=api_key,
                                               requests_per_minute=20)

        return intermediate_outputs


# Summarize.traverse_sub_path('SQuALITY/sparse/min')
# Summarize.traverse_sub_path('SQuALITY/oracle')
Summarize.traverse_sub_path('SQuALITY/dense/min')
# Summarize.traverse_sub_path('SQuALITY/LLM-embedding/min')
