
import argparse
import os
from utils.file_util import read_jsonl, write_jsonl
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

PROMPT_TEMPLATE = """Q: What position does the president of the Untied States hold in the United States Armed Forces?
A: commander-in-chief

Q: When did Turkey join NATO?
A: 1952

Q: Who succeeded Angela Merkel as the chancellor of Germany?
A: Olaf Scholz

Q: When was Sputnik 1 launched into space?
A: 4 October 1957

Q: What genre was the film in which Whitney Houston made her acting debut?
A: romantic thriller

Q: {question}
A:"""


def generate_answers(llm, data):
	sampling_params = SamplingParams(max_tokens=10)
	questions = [PROMPT_TEMPLATE.format(question=sample["question"]) for sample in data]
	generations = llm.generate(questions, sampling_params)
	generated_answers = [gen.outputs[0].text for gen in generations]
	return generated_answers



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
<<<<<<< HEAD
	parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
=======
	parser.add_argument("--model_path", type=str, default="meta-")
>>>>>>> 7f33fde (dpr)
	parser.add_argument("--dataset_path", type=str, default="data/chat_gpt_questions/split_easy.jsonl")
	parser.add_argument("--output_dir", type=str, default="evaluation/eval_out")
	parser.add_argument("--is_adapter", action='store_true')
	args = parser.parse_args()
	data = read_jsonl(args.dataset_path)
	if args.is_adapter:
		peft_model_id = args.model_path
		config = PeftConfig.from_pretrained(peft_model_id)
		inference_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
		tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
		model = PeftModel.from_pretrained(inference_model, peft_model_id)
		model = model.merge_and_unload()
		model.save_pretrained(
                "evaluation/tmp/" + args.model_path.split("/")[-1])
		tokenizer.save_pretrained("evaluation/tmp/" + args.model_path.split("/")[-1])
		args.model_path = "evaluation/tmp/" + args.model_path.split("/")[-1]
	llm = LLM(args.model_path)
	generated_answers = generate_answers(llm, data)
	data_with_answers = [sample | {"generated_answer": ans} for sample, ans in zip(data, generated_answers)]
	output_path = os.path.join(args.output_dir,  "_".join(args.model_path.split("/")[-2:]))
	os.makedirs(output_path, exist_ok=True)
	output_path = os.path.join(output_path, args.dataset_path.split("/")[-1])
	print(f"Writing results to {output_path}")
	write_jsonl(output_path, data_with_answers)
