import os
import json
import argparse
from tqdm import tqdm
from metric_realization import *


def read_fetaqa_reference_jsonl(input_path):
    data = {}
    with open(input_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            example = json.loads(line)
            example_id = str(example["feta_id"])
            data[example_id] = example
    return data


def lower_table(table):
    return table


def convert_table(table_array, table_section_title, table_page_title, highlighted_cell_ids):
    # 获取标题行
    header = table_array[0]

    # 转换表格格式
    converted_table = []
    converted_table.append(['table_page_title', table_page_title.split()])
    converted_table.append(['table_section_title', table_section_title.split()])
    for index in highlighted_cell_ids:
        row = index[0]
        column = index[1]
        value = table_array[row][column]
        attr = header[column]
        converted_table.append([attr.strip().replace(' ', '_'), value.split()])
    return converted_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str,
                        default="../prediction_files/fetaqa/reastap_large_test_predictions.json")
    parser.add_argument("--reference_path", type=str, default="../reference_files/fetaQA-v1_test.jsonl")
    parser.add_argument("--output_path", type=str,
                        default="../evaluation_files/fetaqa/reastap_large_test_evaluation.json")
    parser.add_argument("--max_example", type=int, default=100)

    # TODO: Wencai
    SCORE_FUNCTION_MAP = {
        # "rouge": calculate_rouge,
        # "bleu": calculate_bleu,
        # "meteor": calculate_meteor,
        # "bertscore": calculate_bert_score,
        # "bleurt": calculate_bleurt,
        # "moverscore":calculate_mover_score,
        # "bartscore":calculate_bart_score,
        # "rougewe":calculate_rouge_we,
        "parent": calculate_parent,
    }
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    predictions = json.load(open(args.prediction_path))
    references = read_fetaqa_reference_jsonl(args.reference_path)

    output_data = {}
    for example_id, pred_text in tqdm(predictions.items()):
        output_example = {}
        output_example["reference"] = references[example_id]["answer"].lower()
        output_example["prediction"] = pred_text
        output_example["score"] = {}

        for score_type, score_fn in SCORE_FUNCTION_MAP.items():
            if score_fn != calculate_parent:
                results = score_fn(
                    ref=output_example["reference"], pred=output_example["prediction"])
            else:
                table_array = references[example_id]["table_array"]
                table_page_title = references[example_id]["table_page_title"]
                highlighted_cell_ids = references[example_id]["highlighted_cell_ids"]
                table_section_title = references[example_id]["table_section_title"]
                table_array = convert_table(table_array=table_array, table_page_title=table_page_title,
                                            table_section_title=table_section_title,
                                            highlighted_cell_ids=highlighted_cell_ids)
                # table_array=lower_table(table_array)
                results = score_fn(
                    ref=output_example["reference"], pred=output_example["prediction"], table=table_array)
            for score_name, score_result in results.items():
                output_example["score"][score_name] = score_result
                print(score_name + ":" + str(score_result))

        output_data[example_id] = output_example

    json.dump(output_data, open(args.output_path, "w"), indent=4)
