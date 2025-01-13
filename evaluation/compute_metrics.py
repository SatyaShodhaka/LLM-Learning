from utils.file_util import read_jsonl
from evaluate import load
import os

if __name__ == "__main__":
    model_to_be_evaluated = "Vanilla_lr3e-5_rank8_epoch10"
    # print(model_to_be_evaluated)
    base_path = f"evaluation/eval_out/{model_to_be_evaluated}"
    for res_file in ["split_easy.jsonl", "split_medium.jsonl", "split_hard.jsonl"]:
        print(f"Model: {model_to_be_evaluated}, Split: {res_file}")
        data_path = os.path.join(base_path, res_file)
        data = read_jsonl(data_path)
        exact_match_metric = load("exact_match")
        answer = []
        generated_answer = []

        answer_new = []
        generated_answer_new = []

        answer_old = []
        generated_answer_old = []
        score = 0
        for current_sample in data:
            # for ans, generated_ans in zip(current_sample['answer'], current_sample['generated_answer']):
            cleaned_answer = current_sample['answer'] if current_sample['answer'] is not None else " "
            cleaned_generated_answer = current_sample['generated_answer'] if current_sample['generated_answer'] is not None else " "
            answer.append(cleaned_answer)
            generated_answer.append(cleaned_generated_answer)

            if cleaned_answer in cleaned_generated_answer:
                score += 1

            # if current_sample['label'] == 1:
            #     answer_new.append(current_sample['answer'] if current_sample['answer'] is not None else " ")
            #     generated_answer_new.append(current_sample['generated_answer'] if current_sample['generated_answer'] is not None else " ")
            # else:
            #     answer_old.append(current_sample['answer'] if current_sample['answer'] is not None else " ")
            #     generated_answer_old.append(current_sample['generated_answer'] if current_sample['generated_answer'] is not None else " ")
        # print(generated_answer)
        exact_match_score = exact_match_metric.compute(predictions=generated_answer, references=answer)
        print("Score: ", score/len(data))
        print("Overall EM: ", exact_match_score)

    # exact_match_score = exact_match_metric.compute(predictions=generated_answer_new, references=answer_new)
    # print("New: ", exact_match_score)

    # exact_match_score = exact_match_metric.compute(predictions=generated_answer_old, references=answer_old)
    # print("Old: ", exact_match_score)
    # correct_predictions = sum(1 for actual, predicted in zip(labels, predictions) if actual == predicted)
    # total_predictions = len(annotated_dataset['output'])
    # accuracy = correct_predictions / total_predictions * 100
    # precision = precision_score(labels, predictions)
    # recall = recall_score(labels, predictions)
    # f1 = f1_score(labels, predictions)


    # print("Accuracy: {:.2f}%".format(accuracy))
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1)
