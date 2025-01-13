import argparse
from datasets import load_dataset
import json
import pandas as pd

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--huggingface_dataset", required=True, type=str) # Huggingface dataset name
   parser.add_argument("--dataset_split", type=str) # Huggingface dataset split
   parser.add_argument("--outfile", required=True, type=str) # path to the output file
   args = parser.parse_args()
   return args


def load_and_download_hf_dataset(outfile, huggingface_dataset, dataset_split=None):
   if len(dataset_split) == 0:
      dataset = load_dataset(huggingface_dataset)
   else:
      dataset = load_dataset(huggingface_dataset, split=dataset_split)

   with open(outfile , 'w') as f:
      for element in dataset:
         f.write(json.dumps(element) + "\n")

if __name__ == "__main__":
   args = get_args()
   load_and_download_hf_dataset(
      args.outfile, 
      args.huggingface_dataset,
      args.dataset_split
   )
   