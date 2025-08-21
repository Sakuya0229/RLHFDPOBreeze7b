import os
import torch
import re
import json
import gdown
from datasets import Dataset
import pandas as pd
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, GenerationConfig
from tqdm.auto import tqdm
from trl import DPOTrainer, DPOConfig
# Open and load the json dataset
with open("GenAI_hw6_dataset/labelled_data.json", 'r', encoding="utf-8") as jsonfile:
    full_data = json.load(jsonfile)

with open("GenAI_hw6_dataset/test_prompt.json", 'r', encoding="utf-8") as jsonfile:
    test_data = json.load(jsonfile)
model = AutoModelForCausalLM.from_pretrained(
    'Breeze-7B-Instruct-v0_1',
    device_map='auto',
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
)


tokenizer = AutoTokenizer.from_pretrained('Breeze-7B-Instruct-v0_1')
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token


def data_formulate(data):
    messages = [
        {"role": "system", "content": '回覆請少於20字'},
        {"role": "user", "content": data['prompt']},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return prompt


# Get response from the original model
original_model_response = []
for data in tqdm(test_data):
    id = data['id']
    print(f'Question {id}:\n'+data['prompt'])
    inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=200,
        pad_token_id=tokenizer.pad_token_id
    )
    output = model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[
        0].split('[/INST] ')[1]
    original_model_response.append(output)
    print('Response from original model:\n'+output+'\n')

num_epoch = 3
data_size = 10
support_ratio = 1

# Select part of the data for training
training_data = full_data[:data_size]

# Define the size of the support dataset
support_data_size = int(data_size * support_ratio)

# Prepare the data for the training dataset
prompt_list = [data_formulate(data) for data in training_data]
chosen_list = [data['support'] for data in training_data[:support_data_size]
               ] + [data['oppose'] for data in training_data[support_data_size:]]
rejected_list = [data['oppose'] for data in training_data[:support_data_size]
                 ] + [data['support'] for data in training_data[support_data_size:]]
position_list = ['support' for _ in range(
    support_data_size)] + ['oppose' for _ in range(data_size - support_data_size)]

# Create the training dataset
train_dataset = Dataset.from_dict(
    {'prompt': prompt_list, 'position': position_list, 'chosen': chosen_list, 'rejected': rejected_list})
pd.DataFrame(train_dataset).rename(
    columns={"chosen": "preferred", "rejected": "non-preferred"})

# Training
training_args = DPOConfig(
    output_dir='./output',
    per_device_train_batch_size=1,
    num_train_epochs=num_epoch,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=1,
    warmup_ratio=0.1,
    report_to='none',
    beta=0.1,
    bf16=False,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    # tokenizer=tokenizer,
    peft_config=peft_config,
)
# dpo_trainer.tokenizer = tokenizer
dpo_trainer.train()
# Get response from the trained model
trained_model_response = []
for data in tqdm(test_data):
    id = data['id']
    print(f'Question {id}:\n'+data['prompt'])
    inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=200,
        pad_token_id=tokenizer.pad_token_id
    )
    output = model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[
        0].split('[/INST] ')[1]
    trained_model_response.append(output)
    print('Response from trained model:\n'+output+'\n')


model_response = []
print(
    f'num_epoch: {num_epoch}\ndata_size: {data_size}\nsupport_ratio: {support_ratio}')
print()
for data in test_data:
    id = data['id']
    ref_output = original_model_response[id-1]
    output = trained_model_response[id-1]
    print(f'Question {id}:\n'+data['prompt'])
    print('Response from original model:\n'+ref_output)
    print('Response from trained model:\n'+output)
    print()
    model_response.append({'id': data['id'], 'prompt': data['prompt'],
                          'response_from_original_model': ref_output, 'response_from_trained_model': output})
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
outfile_name = f"epoch-{num_epoch}_size-{data_size}_ratio-{support_ratio}.json"
outfile_path = os.path.join(output_dir, outfile_name)
with open(outfile_path, "w", encoding='UTF-8') as outfile:
    json.dump(model_response, outfile, indent=4, ensure_ascii=False)
