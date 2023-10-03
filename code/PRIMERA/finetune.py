
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from transformers import (
    AutoTokenizer,
    LEDConfig,
    LEDForConditionalGeneration,
)
TOKENIZER = AutoTokenizer.from_pretrained('allenai/PRIMERA')
config=LEDConfig.from_pretrained('allenai/PRIMERA')
MODEL = LEDForConditionalGeneration.from_pretrained('allenai/PRIMERA')
# MODEL = torch.load('model_primera.pth')

MODEL.gradient_checkpointing_enable()
PAD_TOKEN_ID = TOKENIZER.pad_token_id
DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")

# Move the MODEL to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(device)


import os
from datasets import Features, Value, load_dataset


folder_path = "."
# Load the datasets
features = Features({
        'input': Value('string'),  # Assuming the 'input' field in JSON is a string
        'output': Value('string')  # Assuming the 'output' field in JSON is a string
    })

train_file = os.path.join(folder_path, 'train_primera.json')
test_file = os.path.join(folder_path, 'test_primera.json')
train_dataset = load_dataset('json', data_files=train_file ,features=features)['train']
test_dataset = load_dataset('json', data_files=test_file ,features=features)['train']


import re
def process_document(documents):
    input_ids_all=[]
    for data in documents:
        all_docs = re.split(r'\[SEP\]|\<doc-sep\>', data)


        for i, doc in enumerate(all_docs):
            doc = doc.replace("\n", " ")
            doc = " ".join(doc.split())
            all_docs[i] = doc

        #### concat with global attention on doc-sep
        input_ids = []
        for doc in all_docs:
            input_ids.extend(
                TOKENIZER.encode(
                    doc,
                    truncation=True,
                    max_length=4096 // len(all_docs),
                )[1:-1]
            )
            input_ids.append(DOCSEP_TOKEN_ID)
        input_ids = (
            [TOKENIZER.bos_token_id]
            + input_ids
            + [TOKENIZER.eos_token_id]
        )
        input_ids_all.append(torch.tensor(input_ids))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID
    )
    return input_ids




def process_labels(labels):
    label_ids_all=[]
    for label in labels:
        label = label.replace("\n", " ")
        label = " ".join(label.split())
        label_ids = TOKENIZER.encode(
            label,
            truncation=True,
            max_length=1024  # You may adjust this max length depending on your needs
        )
        label_ids = (
            [TOKENIZER.bos_token_id]
            + label_ids[1:-1]
            + [TOKENIZER.eos_token_id]
        )
        label_ids_all.append(torch.tensor(label_ids))
    label_ids = torch.nn.utils.rnn.pad_sequence(
        label_ids_all, batch_first=True, padding_value=TOKENIZER.pad_token_id
    )
    return label_ids

def preprocess_batch(batch):
    input_ids = process_document(batch['input'])
    labels = process_labels(batch['output'])  # assuming you want to preprocess labels in the same way

    # Convert to list for dataset storage
    input_ids = input_ids.tolist()
    labels = labels.tolist()

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': [[1] * len(input_seq) for input_seq in input_ids],  # assuming no padding tokens require masking
    }


# Apply the function to each batch of data
train_dataset = train_dataset.map(preprocess_batch, batched=True, batch_size=16)



# Create a data collator (You can use the default data collator for Seq2Seq tasks)
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=MODEL)

# Setup training arguments and train
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    # warmup_steps=1000,
    # weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=MODEL,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=TOKENIZER,
)

# Fine-tune the model
trainer.train()

MODEL.eval()
def batch_process(batch):
    input_ids = process_document(batch['input'])
    input_ids = input_ids.to(device)

    global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
    global_attention_mask[:, 0] = 1

    generated_ids = MODEL.generate(
        input_ids=input_ids,
        global_attention_mask=global_attention_mask,
        use_cache=True,
        max_length=1024,
        num_beams=5,
    )

    # Free up GPU memory
    del input_ids
    torch.cuda.empty_cache()

    generated_str = TOKENIZER.batch_decode(
        generated_ids.tolist(), skip_special_tokens=True
    )

    result = {'generated_summaries': generated_str}
    return result

torch.cuda.empty_cache()

result = test_dataset.map(batch_process, batched=True, batch_size=4)

import json

# Extracting the 'generated_summaries' from the result
generated_summaries = [item['generated_summaries'] for item in result]

# Saving to JSON file

with open('generated_summaries_new_primera.json', 'w') as f:
    json.dump(generated_summaries, f, indent=4)
