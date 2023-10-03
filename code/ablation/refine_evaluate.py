import json
from PACKAGE import asyncThread, multi_rouge
import os

api_key = ''


class LoadEvaluateData:
    @staticmethod
    def load_pred(path, model_name):
        predictions = []
        # pred_file = model_name + '_intermediate_summary.json'
        pred_file = model_name + '_summary.json'
        file_path = os.path.join(path, 'refine/' + pred_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                predictions = json.load(f)

        # predictions = [item[0] for item in predictions]

        return predictions

    @staticmethod
    def load_ref(path):
        ref_file = 'test.json'
        references = []
        file_path = os.path.join(path, ref_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                references = json.load(f)
            references = [
                data_item['Summary']
                for data_item in references]
        return references


class Evaluate:
    @staticmethod
    def squality_rouge(path, predictions, references, model_name):
        # Calculate
        print('Evaluate rouge score (use squality)')
        rouge_object = multi_rouge.Rouge()
        squality_rouge_score = rouge_object._compute(predictions=predictions,
                                                     references=[[item] for item in references], use_stemmer=True)
        # Save
        file_name = model_name + '_squality_rouge.json'
        file_path = os.path.join(path, 'refine/' + file_name)
        with open(file_path, 'w') as f:
            f.write(str(squality_rouge_score))

    @staticmethod
    def gpt_eval(path, predictions, references, model_name):
        # Get prompt
        # metric_list = ['coh', 'con', 'flu', 'rel']
        metric_list = ['con', 'rel']
        for metric_type in metric_list:
            prompt = open('GPTeval/prompts/' + metric_type + '_detailed.txt').read()

            # Get messages
            messages = []
            for index, prediction in enumerate(predictions):
                reference = references[index]
                cur_prompt = prompt.replace('{{Document}}', reference).replace('{{Summary}}', prediction)
                messages.append([{"role": "system", "content": cur_prompt}])

            response_13list = []
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

                response_13list.extend(response_list)

            # Calaulate Average
            average = [sum(response_13list) / len(response_13list)]

            # Save
            # save_path = os.path.join(path, 'refine_l2h/' + model_name + '_truncate_' + metric_type + '_gpteval.json')
            save_path = os.path.join(path, 'refine/' + model_name + '_' + metric_type + '_gpteval.json')
            gpteval = {'Summary': response_13list, 'average': average}
            with open(save_path, 'w') as f:
                temp = json.dumps(gpteval)
                f.write(temp)

    @staticmethod
    def evaluate(path, model_name, bert=False, rouge=False, another_rouge=False, bleurt=False, gpteval=False):
        # Load predictions
        predictions = LoadEvaluateData.load_pred(path, model_name)
        if not predictions:
            return
        # Load references
        references = LoadEvaluateData.load_ref(path)
        # Load random index
        with open('QMSum/randomIndex/index.json', 'r') as f:
            random_index_list = json.load(f)
        # Change references same to prediciton
        if model_name.startswith('gpt3') or model_name.startswith('gpt4'):
            references = [references[index] for index in random_index_list]
        # predictions = [predictions[index] for index in random_index_list]

        # Delete empty
        references = [references[index] for index, item in enumerate(predictions) if item != '']
        predictions = [predictions[index] for index, item in enumerate(predictions) if item != '']

        if rouge:
            Evaluate.squality_rouge(path, predictions, references, model_name)
        if another_rouge:
            Evaluate.another_rouge(path, predictions, references, model_name)
        if bert:
            Evaluate.bert(path, predictions, references, model_name)
        if bleurt:
            Evaluate.bleurt(path, predictions, references, model_name)
        if gpteval:
            Evaluate.gpt_eval(path, predictions, references, model_name)

    @staticmethod
    def traverse_path(root, model_name, bert=False, rouge=False, another_rouge=False, bleurt=False, gpteval=False):
        start_flag = True
        for path, dirs, files in os.walk(root):
            if files and dirs:
                if start_flag:
                    start_flag = False
                else:
                    print("sleep 20 seconds")
                    # time.sleep(20)
                print(f'prepare {path}')
                Evaluate.evaluate(path, model_name, bert, rouge, another_rouge, bleurt, gpteval)
                print(f'write to {path}')


Evaluate.evaluate(path='QMSum/sparse/MIN', model_name='gpt3', rouge=True, gpteval=True)
