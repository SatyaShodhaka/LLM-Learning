import time

import pandas as pd
from scraper import scrape_webpage
from file_util import write_jsonl

if __name__ == "__main__":
    df = pd.read_csv("data/freshqa/FreshQA_v11132023 - freshqa.csv")
    # year_condition = df['effective_year'] == '2023'
    # fact_type_condition = df['fact_type'].isin(['slow-changing', 'fast-changing'])
    # one_hop_condition = df['num_hops'] == 'one-hop'
    # not_false_premise_condition = df['false_premise'] == False
    # final_condition = fact_type_condition & one_hop_condition & not_false_premise_condition
    # Apply the filter to the DataFrame
    # filtered_df = df[final_condition]
    filtered_df = df
    filtered_df = filtered_df.dropna(subset=['source'])
    print(filtered_df.shape)
    data = []
    sample_count = 1
    for idx, row in filtered_df.iterrows():
        print(f"Processing: {sample_count}")
        sample = {}
        sample['freshqa_idx'] = row['id']
        sample['freshqa_split'] = row['split']
        sample['freshqa_next_review'] = row['next_review']
        sample['freshqa_fact_type'] = row['fact_type']
        sample['question'] = row['question']
        sample['answer'] = row['answer_0']
        sample['context_url'] = row['source']
        sample['effective_year'] = row['effective_year']
        text = scrape_webpage(row["source"])
        if text is None:
            print(f"Unable to scrape row id: {row['id']}")
            continue
        else:
            sample['context'] = text
        sample_count += 1
        data.append(sample)
        time.sleep(2) # gentle scraping
    write_jsonl("data/freshqa/freshqa_extracted_data_all.jsonl", data)
    