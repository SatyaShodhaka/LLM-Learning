import pandas as pd
from file_util import write_jsonl

def join_jsonl_files(file1, file2, field1, field2):
    # Read JSONL files into Pandas DataFrames
    df1 = pd.read_json(file1, lines=True)
    df2 = pd.read_json(file2, lines=True)
    df2 = df2.drop(['question', 'answer'], axis=1)

    # Merge DataFrames based on the specified field
    merged_df = pd.concat([df1, df2], axis=1)

    # Filter the merged DataFrame based on the provided question

    assert len(merged_df) == len(df2)
    print(merged_df.head())
    return merged_df.to_dict(orient='records')

if __name__ == "__main__":
    file1 = "data/chat_gpt_questions/wikimia.jsonl"
    file2 = "data/cpt_results/wikimia.jsonl"
    data = join_jsonl_files(file1, file2, "question", "question")
    write_jsonl("data/cpt_results/wikimia_10_epochs_with_all_fields.jsonl", data)


