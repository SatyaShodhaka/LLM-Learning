from datasets import load_dataset
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
from accelerate import Accelerator
import argparse
from functools import partial

def load_data(dataset_name):
    if dataset_name=="wikimia":
        dataset = load_dataset("swj0419/WikiMIA", split="WikiMIA_length32")
        dataset = dataset.filter(lambda x: x["label"]==0)
    elif dataset_name=="temporal_wiki":
        dataset = load_dataset(
            "json",
            data_files='/data/RAG/new_data_whole.jsonl',
            split='train'
        )
    elif dataset_name == "temporal_wiki_CLeasy":
        dataset = load_dataset(
            "json", 
            data_files='/cl_scoring/detect-pretrain-code/results/split_easy.jsonl', 
            split='train'
        )
    elif dataset_name == "temporal_wiki_CLmedium":
        dataset = load_dataset(
            "json", 
            data_files='/cl_scoring/detect-pretrain-code/results/split_medium.jsonl', 
            split='train'
        )
    elif dataset_name == "temporal_wiki_CLhard":
        dataset = load_dataset(
            "json", 
            data_files='/cl_scoring/detect-pretrain-code/results/split_hard.jsonl', 
            split='train'
        )
    return dataset



def preprocess_function(examples, tokenizer):
    return tokenizer(examples["context"])

def group_texts(examples):
    block_size = 512
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(args):
    dataset = load_data(args.dataset_name)
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"

    tokenized_dataset = dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
    )
    
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        save_strategy="no",
        num_train_epochs=1,
        push_to_hub=False,
        do_eval=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        eval_dataset=lm_dataset
    )
    accelerator = Accelerator()
    
    
    trainer.train()
    
    model.save_pretrained(
      args.output_dir,
      is_main_process=accelerator.is_main_process,
      save_function=accelerator.save,
      state_dict=accelerator.get_state_dict(model, unwrap=False),
    )
    tokenizer.save_pretrained(args.output_dir)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--dataset_name", type=str, default="temporal_wiki")
    parser.add_argument("--output_dir", type=str, default="pythia-1.4b-pytorch-temporal-wiki")
    args = parser.parse_args()
    main(args)



'''eli5 = load_dataset("eli5", split="train_asks[:5000]")

eli5 = eli5.train_test_split(test_size=0.2)

eli5 = eli5.flatten()



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.padding_side = "right"

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

block_size = 128




lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

compute_dtype = getattr(torch, "float16")


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16,use_flash_attention_2=True)
model.config.use_cache = False
model.config.pretraining_tp = 1

training_args = TrainingArguments(
    output_dir="my_awesome_eli5_clm-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    num_train_epochs=1,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)
#accelerator = Accelerator()
#model, lm_dataset, trainer =  accelerator.prepare(model, lm_dataset, trainer)
trainer.train()'''

