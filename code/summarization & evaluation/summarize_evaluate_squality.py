import json
import time
# import pygame
from PACKAGE import asyncThread, metric_realization, multi_rouge
import os
import bert_score
from tqdm import tqdm


def load_api_key(index):
    with open('../../keys.json', 'r') as f:
        api_keys = json.load(f)
    return api_keys[index]


# 0-out
# 1-out
# 2-out
# 3-out
api_key = ''


class Summarize:
    @staticmethod
    def gpt_run(user_list, requests_per_minute):
        system = [
            '''Imagine you are a human annotator. You will receive a query and a article.Read the article and answer the question in about 200-400 words.The question may be a specific question or a general question which ask you to summarize the storyline. Both kinds are all answerable. Please read the article carefully.''',
            '''You are a helpful assistant that gives long answer to question based on a long story.''',
            '''You are a helpful assistant that gives long answer to question based on a long meeting.''']
        # messages = [[{"role": "user", "content": f"print the number {i}"}] for i in range(100)]
        messages = [[{"role": "system", "content": system[1]},
                     # {"role": "user", "content": user1},
                     # {"role": "assistant", "content": assistant1},
                     {"role": "user", "content": user}] for user in user_list]
        response_list = asyncThread.run(messages=messages,
                                        engine_name="gpt-3.5-turbo-16k-0613",
                                        temperature=0.7,
                                        max_tokens=600,
                                        top_p=0.9,
                                        api_key=api_key,
                                        requests_per_minute=requests_per_minute)
        return response_list

    @staticmethod
    def get_input(data):
        input_user_list = []
        for item in data:
            query = item['Query']
            article = item['Article']
            input_string = f"Write an answer based on the following question and the story.\n QUESTION:{query}\n STORY:{article}\n SUMMARY: \n"
            input_user_list.append(input_string)
        return input_user_list

    @staticmethod
    def traverse_summary(folder_path):
        # folder_path 指定要遍历的文件夹路径
        # folder_path = "SQuALITY/sparse/min"
        wait_process_files = ['max.json', 'mean.json', 'min.json', 'Academic.json', 'Committee.json', 'Product.json',
                              'dev.json', 'test.json', 'train.json']
        for root, dirs, files in os.walk(folder_path):
            # 遍历当前目录下的文件夹
            for dir_name in dirs:
                print("文件夹：", os.path.join(root, dir_name))
            # 遍历当前目录下的文件
            for file_name in files:
                # If it is origin file
                if file_name not in wait_process_files:
                    continue

                # Set requests_per_minute
                requests_per_minute = 20
                if root.endswith('min'):
                    requests_per_minute = 40
                # Load data
                with open(os.path.join(root, file_name), 'r') as f:
                    data = json.load(f)
                # Get input
                input_user_list = Summarize.get_input(data)
                # Get response
                new_summary = Summarize.gpt_run(input_user_list, requests_per_minute)
                # Write response
                with open(root + '/summary/newSummary_' + file_name, 'w') as f:
                    temp = json.dumps(new_summary, indent=4)
                    f.write(temp)


class LoadEvaluateData:
    @staticmethod
    def load_pred(path, model_name):
        predictions = []
        pred_file = model_name + '_summary.json'
        file_path = os.path.join(path, 'summary/' + pred_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                predictions = json.load(f)
        return predictions

    @staticmethod
    def load_ref(path, dataset: str):
        ref_file = 'test.json'
        references = []
        file_path = os.path.join(path, ref_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                references = json.load(f)
            if dataset.lower() == 'squality':
                references = [
                    [data_item['Summary_1'], data_item['Summary_2'], data_item['Summary_3'], data_item['Summary_4']] for
                    data_item in references]
            elif dataset.lower() == 'qmsum':
                references = [
                    data_item['Summary']
                    for data_item in references]
        return references


class Evaluate:

    @staticmethod
    def squality_rouge(path, predictions, references, model_name, dataset):
        # Calculate
        print('Evaluate rouge score (use squality)')
        rouge_object = multi_rouge.Rouge()
        squality_rouge_score = []

        if dataset.lower() == 'squality':
            squality_rouge_score = rouge_object._compute(predictions=predictions, references=references,
                                                         use_stemmer=True)
        elif dataset.lower() == 'qmsum':
            squality_rouge_score = rouge_object._compute(predictions=predictions,
                                                         references=[[item] for item in references], use_stemmer=True)
        # Save
        file_name = model_name + '_squality_rouge.json'
        file_path = os.path.join(path, 'evaluation/' + file_name)
        with open(file_path, 'w') as f:
            f.write(str(squality_rouge_score))

    @staticmethod
    def bert(path, predictions, references, model_name):
        print('Evaluate bert score')
        # 批次大小
        batch_size = 261
        bert_scores = {'p': [], 'r': [], 'f1': [], 'average_p': 0, 'average_r': 0, 'average_f1': 0}
        num_batches = (len(predictions) + batch_size - 1) // batch_size  # 计算需要的批次数量

        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min(start + batch_size, len(predictions))

            pred_batch = predictions[start:end]
            ref_batch = references[start:end]

            p, r, f1 = bert_score.score(pred_batch, ref_batch, lang='en')
            # Add in bert_scores
            for index in range(len(p)):
                bert_scores['r'].append(float(p[index]))
                bert_scores['p'].append(float(r[index]))
                bert_scores['f1'].append(float(f1[index]))

        # Calculate average bert
        average_p = sum(bert_scores['p']) / len(bert_scores['p'])
        average_r = sum(bert_scores['r']) / len(bert_scores['r'])
        average_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
        bert_scores['average_p'] = average_p
        bert_scores['average_r'] = average_r
        bert_scores['average_f1'] = average_f1
        # Save
        file_name = model_name + '_bert_score.json'
        file_path = os.path.join(path, 'evaluation/' + file_name)
        with open(file_path, 'w') as f:
            temp = json.dumps(bert_scores)
            f.write(temp)

    @staticmethod
    def another_rouge(path, predictions, references, model_name):
        print('Evaluate rouge score (use another way)')
        rouge_score = metric_realization.calculate_rouge(ref=references, pred=predictions)
        # Save
        file_name = model_name + '_rouge.json'
        file_path = os.path.join(path, 'evaluation/' + file_name)
        with open(file_path, 'w') as f:
            temp = json.dumps(rouge_score)
            f.write(temp)

    @staticmethod
    def bleurt(path, predictions, references, model_name):
        print('Evaluate bleurt score')
        rouge_score = metric_realization.calculate_bert_score(ref=references, pred=predictions)
        # Save
        file_name = model_name + '_bleurt.json'
        file_path = os.path.join(path, 'evaluation/' + file_name)
        with open(file_path, 'w') as f:
            temp = json.dumps(rouge_score)
            f.write(temp)

    @staticmethod
    def gpt_eval(path, predictions, references, model_name, dataset: str, gpteval_summary_index=1):

        # metric_list = ['coh', 'con', 'flu', 'rel']
        # Get prompt
        metric_list = ['con', 'rel']
        for metric_type in metric_list:
            prompt = open('GPTeval/prompts/' + metric_type + '_detailed.txt').read()

            # Get messages
            messages = []
            for index, prediction in enumerate(predictions):
                reference = references[index]
                cur_prompt = prompt.replace('{{Document}}', reference).replace('{{Summary}}', prediction)
                messages.append([{"role": "system", "content": cur_prompt}])

            response_total_list = []
            for _ in range(23):
                print(_)
                # Send request
                response_list = asyncThread.run(messages=messages,
                                                engine_name="gpt-3.5-turbo-16k-0613",
                                                # engine_name="gpt-4-0613",
                                                temperature=1,
                                                max_tokens=5,
                                                top_p=1,
                                                api_key=api_key,
                                                requests_per_minute=180)

                # Del non-numeric
                num_list = ['1', '2', '3', '4', '5']
                response_list = [item for item in response_list if item and item[0] in num_list]
                response_list = [int(item[0]) for item in response_list]

                response_total_list.extend(response_list)

            # Calaulate Average
            average = [sum(response_total_list) / len(response_total_list)]

            # Save
            # save_path = os.path.join(path, 'evaluation/' + model_name + '_truncate_' + metric_type + '_gpteval.json')
            save_path = os.path.join(path, 'evaluation/' + model_name + '_' + metric_type + '_gpteval.json')

            if dataset.lower() == 'squality':
                # Load fore gpteval
                if os.path.exists(save_path):
                    with open(save_path, "r") as f:
                        gpteval = json.load(f)
                else:
                    gpteval = {}
                gpteval['Summary_' + str(gpteval_summary_index)] = response_total_list
                gpteval['average'] = average
                with open(save_path, 'w') as f:
                    temp = json.dumps(gpteval)
                    f.write(temp)
            elif dataset.lower() == 'qmsum':
                gpteval = {'Summary': response_total_list, 'average': average}
                with open(save_path, 'w') as f:
                    temp = json.dumps(gpteval)
                    f.write(temp)

    @staticmethod
    def evaluate(path, model_name, dataset: str, bert=False, rouge=False, another_rouge=False, bleurt=False,
                 gpteval=False, gpteval_summary_index=1):
        # Load predictions
        predictions = LoadEvaluateData.load_pred(path, model_name)
        if not predictions:
            return
        # Load references
        references = LoadEvaluateData.load_ref(path, dataset)
        # Load random index
        with open('QMSum/randomIndex/index.json', 'r') as f:
            random_index_list = json.load(f)
        # Change references same to prediciton
        if model_name.startswith('gpt4'):
            references = [references[index] for index in random_index_list]
        elif dataset == 'qmsum' and model_name.startswith('gpt3'):
            references = [references[index] for index in random_index_list]
        # Delete empty
        references = [references[index] for index, item in enumerate(predictions) if item != '']
        predictions = [predictions[index] for index, item in enumerate(predictions) if item != '']

        if rouge:
            Evaluate.squality_rouge(path, predictions, references, model_name, dataset)
        if another_rouge:
            Evaluate.another_rouge(path, predictions, references, model_name)
        if bert:
            Evaluate.bert(path, predictions, references, model_name)
        if bleurt:
            Evaluate.bleurt(path, predictions, references, model_name)
        if gpteval:
            if dataset.lower() == 'squality':
                references = [item[gpteval_summary_index - 1] for item in references]
            Evaluate.gpt_eval(path, predictions, references, model_name, dataset, gpteval_summary_index)

    @staticmethod
    def traverse_path(root, model_name, dataset, bert=False, rouge=False, another_rouge=False, bleurt=False,
                      gpteval=False, gpteval_summary_index=1):
        start_flag = True
        for path, dirs, files in os.walk(root):
            if files and dirs:
                if start_flag:
                    start_flag = False
                else:
                    print("sleep 45 seconds")
                    # time.sleep(45)
                print(f'prepare {path}')
                Evaluate.evaluate(path, model_name, dataset, bert, rouge, another_rouge, bleurt, gpteval,
                                  gpteval_summary_index)
                print(f'write to {path}')

# 初始化所有 Pygame 模块
# pygame.init()
# waiting to process: dense LLM-embedding oracle sparse —— bart rel

# Play sound when done
# pygame.mixer.music.load("雷达铃声.mp3")
# pygame.mixer.music.set_volume(0.1)
# for _ in range(3):
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():
#         continue
