from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
from accelerate import Accelerator
import argparse
from functools import partial
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM, get_peft_model
from tqdm import tqdm
import copy


def load_data(dataset_name):
    if dataset_name=="wikimia":
        dataset = load_dataset("swj0419/WikiMIA", split="WikiMIA_length32")
        dataset = dataset.filter(lambda x: x["label"]==0)
    elif dataset_name=="temporal_wiki":
        dataset = load_dataset("json",data_files='/data/RAG/new_data_whole.jsonl',split='train')
    elif dataset_name=="temporal_wiki_10k":
        dataset = load_dataset("json",data_files='/data/temporal_wiki/new_10k_documents_2500_max_length.jsonl',split='train')
    elif dataset_name == "temporal_wiki_gradient":
        dataset = load_dataset(
            "json", 
            data_files='/cl_scoring/detect-pretrain-code/results/merged.jsonl', 
            split='train'
        )    
    return dataset



def preprocess_function(examples, tokenizer):
    return tokenizer(examples["context"])

def group_texts(examples):
    block_size = 512
    #concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    #total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    # if total_length >= block_size:
    #     total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
     
    total_length = len(examples["input_ids"])
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in examples.items()
    }
    #result["labels"] = result["input_ids"].copy()
    return result



def main(args):
    dataset = load_data(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"

    tokenized_dataset = dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=False,
    )

    input_ids = []
    attention_mask = []
    labels = []
    for i in tqdm(range(len(lm_dataset))):
        for j in range(len(lm_dataset[i]["input_ids"])):
            input_ids.append(lm_dataset[i]["input_ids"][j])
            attention_mask.append(lm_dataset[i]["attention_mask"][j])
            #labels.append(lm_dataset[i]["labels"][j])
    #lm_dataset = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    lm_dataset = {"input_ids": input_ids, "attention_mask": attention_mask}
    lm_dataset = Dataset.from_dict(lm_dataset)

    # lm_dataset = tokenized_dataset
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16,use_flash_attention_2=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # peft_config = LoraConfig(
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     r=8,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, peft_config)

    accelerator = Accelerator()
    
    
    model, lm_dataset = accelerator.prepare(model, lm_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    dataloader = DataLoader(lm_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

    num_epochs = 10
    gradient_accumulation_steps = 64

    for epoch in tqdm(range(num_epochs)):
        #print(model.print_trainable_parameters())
        # accumulated_loss = 0.0
        total_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()

            # Accumulate gradients
            loss = loss / gradient_accumulation_steps
            # accumulated_loss += loss.item()
            accelerator.backward(loss)

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Backward and optimizer step only after 'gradient_accumulation_steps' iterations
                # accelerator.backward(accumulated_loss)
                optimizer.step()
                optimizer.zero_grad()
                # accumulated_loss = 0.0
            
            del inputs
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        #save the model after epoch%3==0
        # if epoch%3==0:
        #     peft_model = copy.deepcopy(model)
        #     peft_model = peft_model.merge_and_unload()
        #     peft_model.save_pretrained(
        #         args.output_dir+str(epoch),
        #         is_main_process=accelerator.is_main_process,
        #         save_function=accelerator.save,
        #         state_dict=accelerator.get_state_dict(model, unwrap=False),
        #     )
        #     del peft_model
        #     gc.collect()
        #     torch.cuda.empty_cache()
            

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    #model = model.merge_and_unload()
    model.save_pretrained(
      args.output_dir,
      is_main_process=accelerator.is_main_process,
      save_function=accelerator.save,
      state_dict=accelerator.get_state_dict(model, unwrap=False),
    )
    torch.cuda.empty_cache()
    tokenizer.save_pretrained(args.output_dir)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--dataset_name", type=str, default="temporal_wiki_10k")
    parser.add_argument("--output_dir", type=str, default="pythia-1.4b-pytorch-temporal-wiki")
    args = parser.parse_args()
    main(args)



