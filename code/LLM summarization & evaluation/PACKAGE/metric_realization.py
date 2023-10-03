import evaluate


def calculate_rouge(ref, pred):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=pred, references=ref)
    # del results["rougeLsum"]

    return results


def calculate_bleu(ref, pred):
    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=[pred], references=[ref])
    del results['precisions']
    del results['brevity_penalty']
    del results['length_ratio']
    del results['translation_length']
    del results['reference_length']

    return results


def calculate_meteor(ref, pred):
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=[pred], references=[ref])

    return results


def calculate_bert_score(ref, pred):
    bertscore = evaluate.load('bertscore')
    results = bertscore.compute(predictions=pred, references=ref, lang='en')
    del results['hashcode']
    # new_result = {'bertscore_p': results['precision'][0], 'bertscore_r': results['recall'][0],
    #               'bertscore_f': results['f1'][0]}
    return results


def calculate_bleurt(ref, pred):
    bleurt = evaluate.load('bleurt')
    results = bleurt.compute(predictions=pred, references=ref)
    new_result = {'bleurt': results['scores'][0]}
    return new_result

# def calculate_mover_score(ref, pred):
#     idf_dict_hyp = moverscore.get_idf_dict(pred)
#     idf_dict_ref = moverscore.get_idf_dict(ref)
#     results = moverscore.word_mover_score([ref], [pred], idf_dict_ref, idf_dict_hyp,
#                                           device=torch.device("cpu"))
#
#     return {"moverscore":results[0]}


def calculate_parent(ref, pred, table):
    return {}
