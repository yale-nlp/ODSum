import json
import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from langchain import HuggingFacePipeline, PromptTemplate,  LLMChain


# Constants and model setup
model_path = "/vast/work/public/ml-datasets/llama-2/Llama-2-70b-chat-hf"
B_INST, E_INST = "<s>[INST] ", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\\n", "\\n<</SYS>>\\n\\n"
SYSTEM_PROMPT = "You are a helpful assistant. After reading a lengthy story, provide a 200 to 300 words answer to a question posed about the story. Directly respond to the question and do not chat with the user."


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id  # open-ended generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)


def load_data_from_path(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_prompt(instruction, system_prompt=SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def get_prompt_no_chat(instruction, system_prompt=SYSTEM_PROMPT):
    prompt_template =  system_prompt + '\n' + instruction 
    return prompt_template

def cut_off_text(text, substr):
    index = text.find(substr)
    if index != -1:
        return text[:index]
    else:
        return text

def generate_and_save_summaries(llm,data, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = []
    for item in data:
        query = item['Query']
        article = item['Article']
 
        formatted_instruction = f"Answer the following question posed about the story.\nQUESTION: {{query}}\nSTORY: {{article}}\nANSWER: "
    
        template = get_prompt(formatted_instruction) if "chat" in model_path else get_prompt_no_chat(formatted_instruction)
     
        print(template)
        prompt = PromptTemplate(template=template, input_variables=["article", "query"])

        llm_chain = LLMChain(prompt=prompt, llm=llm)

        output = llm_chain.run(article=article, query=query)
        results.append(output)

        with open(os.path.join(output_folder, "generated_summaries.json"), 'w') as f:
            json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Run LlaMA for text summarization')
    parser.add_argument('--source_file', type=str, required=True, help='Path to the input data file (test.json)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top_p for generation')
    parser.add_argument('--max_new_tokens', type=int, default=640, help='Maximum target length for generation')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated summaries')
    parser.add_argument('--overwrite_output_dir', action='store_true', help='Overwrite the content of the output directory')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif args.overwrite_output_dir:
        print(f"Overwriting contents in {args.output_dir}")

    data = load_data_from_path(args.source_file)

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )

    print(pipe)

    llm = HuggingFacePipeline(pipeline=pipe)
    generate_and_save_summaries(llm, data, args.output_dir)

if __name__ == "__main__":
    main()
