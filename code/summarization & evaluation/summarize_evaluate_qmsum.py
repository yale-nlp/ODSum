import json
from PACKAGE import asyncThread, metric_realization, multi_rouge
import os
import bert_score
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
import time
import random


def load_api_key(index):
    with open('../../keys.json', 'r') as f:
        api_keys = json.load(f)
    return api_keys[index]


api_key = ''
llm = ChatOpenAI(model_name="gpt-4-0613", openai_api_key=api_key, temperature=0.7, max_tokens=600)


class Split:
    @staticmethod
    def split_string(string, num_segments):
        # 计算每段的大致大小
        chunk_size = 1 + len(string) // num_segments
        chunks = []
        start = 0

        for _ in range(num_segments):
            end = start + chunk_size

            # 如果已经到达字符串尾部，直接结束循环
            if end >= len(string):
                chunks.append(string[start:])
                break

            # 查找最近的可能的句子结尾
            while end < len(string):
                if string[end] in '.!?' and (end + 1 >= len(string) or string[end + 1].isspace()):
                    break
                end += 1

            # 如果找到可能的句子结尾，包括该字符在内
            if end < len(string):
                end += 1

            # 截取从start到end的子串，并添加到chunks列表中
            chunks.append(string[start:end])

            # 更新下一次迭代的起始位置
            start = end

        return chunks

    @staticmethod
    def split_meeting(articles):

        tokens = [llm.get_num_tokens(article) for article in articles]

        split_counts = [1 + token // 6800 for token in tokens]

        split_meetings = []
        for index, article in enumerate(articles):
            split_meetings.append(Summarize.split_string(article, split_counts[index]))
        return tokens, split_meetings

    @staticmethod
    def make_split_meeting_files():
        path = 'SQuALITY'
        wait_process_files = ['test.json']
        for root, dirs, files in os.walk(path):
            # 遍历当前目录下的文件
            for file_name in files:
                # If it is origin file
                if file_name not in wait_process_files:
                    continue

                # 检测有没有创建split文件夹
                if not os.path.lexists(os.path.join(root, 'split')):
                    os.makedirs(os.path.join(root, 'split'))

                # Load data
                with open(os.path.join(root, file_name), 'r') as f:
                    data = json.load(f)
                    articles = [item['Article'] for item in data]
                tokens, split_meetings = Summarize.split_meeting(articles)

                # Write meetings
                with open(os.path.join(root, 'split/split8000_' + file_name), 'w') as f:
                    print(root)
                    temp = json.dumps(split_meetings, indent=4)
                    f.write(temp)


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
        save_final_output = []

        for index, article in enumerate(articles):
            # 运行并处理
            query = queries[index]
            intermediate_outputs = Summarize.intermediate_summary(query, article)
            final_output = Summarize.final_summary(query, intermediate_outputs)
            save_intermediate_outputs.append(intermediate_outputs)
            save_final_output.append(final_output)
            with open(os.path.join(path, 'summary/gpt311_intermediate_summary.json'), 'w') as f:
                temp = json.dumps(save_intermediate_outputs, indent=4)
                f.write(temp)

            with open(os.path.join(path, 'summary/gpt311_summary.json'), 'w') as f:
                temp = json.dumps(save_final_output, indent=4)
                f.write(temp)

    @staticmethod
    def intermediate_summary(query, docs):
        map_prompts = [
            f"Write an answer based on the following question and the given meeting.Try to answer thoroughly and do not leave out useful information.\n MEETING:{doc}\n QUESTION:{query}\n SUMMARY: \n"
            for doc in docs]
        # map_prompts = [
        #     f"Abstract the paragraph from the meeting which can be used to answer the question. Do not leave out useful information.\n MEETING:{doc}\n QUESTION:{query}\n ABSTRACTED PARAGRAPH: \n"
        #     for doc in docs]
        system = "You are a helpful assistant that gives long answer to question based on a long meeting."
        messages = [[{"role": "system", "content": system},
                     {"role": "user", "content": map_prompt}] for map_prompt in map_prompts]
        intermediate_outputs = asyncThread.run(messages=messages,
                                               engine_name="gpt-3.5-turbo-16k-0613",
                                               # engine_name="gpt-4-0613",
                                               temperature=0.7,
                                               max_tokens=600,
                                               top_p=0.9,
                                               api_key=api_key,
                                               requests_per_minute=20)

        return intermediate_outputs

    @staticmethod
    def refine_summary(query, docs, l2h=False):
        if l2h:
            docs.reverse()
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

    @staticmethod
    def final_summary(query, intermediate_outputs):
        combine_prompt = """
        Combine the information of the following text together to form a long passage answering the question.
        Try to answer thoroughly and do not leave out useful information.
        QUESTION:
        "{question}"
        TEXT:
        {text}
        SUMMARY:
        """

        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "question"])
        combine_llm_chain = LLMChain(llm=llm, prompt=combine_prompt_template)

        feed_text = "\n".join(intermediate_outputs)

        output = combine_llm_chain.run({
            'text': feed_text,
            'question': query
        })
        return output


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



