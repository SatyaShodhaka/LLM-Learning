from utils.file_util import read_jsonl
from evaluate import load
import os
from sklearn.metrics import accuracy_score

def constrained_exact_match(predictions, references, constraint):
    matches = 0
    for pred, ref in zip(predictions, references):
        if constraint(pred, ref):
            matches += 1
    return matches / len(references)

def sample_constraint(pred, ref):
    # Example constraint: case-insensitive match
    return ref.lower() in pred.lower()

if __name__ == "__main__":
    model_to_be_evaluated = "Vanilla_lr3e-5_rank8_epoch10"
    base_path = f"evaluation/eval_out/{model_to_be_evaluated}"
    
    for res_file in ["split_easy.jsonl", "split_medium.jsonl", "split_hard.jsonl"]:
        print(f"Model: {model_to_be_evaluated}, Split: {res_file}")
        data_path = os.path.join(base_path, res_file)
        data = read_jsonl(data_path)
        
        exact_match_metric = load("exact_match")
        answer = []
        generated_answer = []
        mcq_references = []
        mcq_predictions = []
        completion_generated = []
        completion_answer = []
        
        for current_sample in data:
            cleaned_answer = current_sample['answer'] if current_sample['answer'] is not None else " "
            cleaned_generated_answer = current_sample['generated_answer'] if current_sample['generated_answer'] is not None else " "
            answer.append(cleaned_answer)
            generated_answer.append(cleaned_generated_answer)
            
            if 'mcq_reference' in current_sample:
                mcq_references.append(current_sample['mcq_reference'])
                mcq_predictions.append(current_sample['mcq_prediction'])
            
            if 'completion_answer' in current_sample:
                completion_answer.append(current_sample['completion_answer'])
                completion_generated.append(current_sample['completion_generated'])
        
        exact_match_score = exact_match_metric.compute(predictions=generated_answer, references=answer)
        cem_score = constrained_exact_match(generated_answer, answer, sample_constraint)
        mcq_accuracy = accuracy_score(mcq_references, mcq_predictions)
        
        if completion_generated and completion_answer:
            completion_em_score = exact_match_metric.compute(predictions=completion_generated, references=completion_answer)
            completion_cem_score = constrained_exact_match(completion_generated, completion_answer, sample_constraint)
        else:
            completion_em_score = 0
            completion_cem_score = 0
        
        print("Overall EM: ", exact_match_score)
        print("Overall CEM: ", cem_score)
        print("MCQ Accuracy: ", mcq_accuracy)
        print("Completion EM: ", completion_em_score)
        print("Completion CEM: ", completion_cem_score)