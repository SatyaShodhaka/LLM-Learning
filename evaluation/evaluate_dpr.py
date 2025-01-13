import numpy as np
from utils import file_util
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

THRESHOLD = 80
NUM_PASSAGES_FOR_EVAL = 1

def count_elements_less_than(lst, number):
    np_array = np.array(lst)
    count = np.sum(np_array < number)
    return count

def read_golden_annotations(file_path):
    df = pd.read_csv(file_path)
    return df['output']

if __name__ == "__main__":
    dpr_results = file_util.read_json("question2passage-dpr/results/FreshQA/freshqa_all_msmarco_results.jsonl")
    gold_truth = read_golden_annotations("data/newwiki_passages.csv")
    final_outputs = []
    for dpr_res in dpr_results:
        current_scores = []
        for idx in range(NUM_PASSAGES_FOR_EVAL):
            current_scores.append(float(dpr_res['ctxs'][idx]['score']))
        # score is lesser than threshold implies less similar
        avg_score = np.mean(current_scores)
        if avg_score < THRESHOLD:
            final_outputs.append(1)
        else:
            final_outputs.append(0)

    correct_predictions = np.sum(final_outputs == gold_truth)
    total_elements = len(final_outputs)

    accuracy = (correct_predictions / total_elements) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    precision = precision_score(gold_truth, final_outputs)

    # Calculate recall
    recall = recall_score(gold_truth, final_outputs)

    # Calculate F1-score
    f1 = f1_score(gold_truth, final_outputs)
    report = classification_report(gold_truth, final_outputs)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    print(report)
