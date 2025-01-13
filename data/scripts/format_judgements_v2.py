import argparse
import json
import pandas as pd

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--judgements_file", required=True, type=str) # path to judgements file in jsonl
   parser.add_argument("--outfile", required=True, type=str) # path to the output file
   args = parser.parse_args()
   return args


def format_judgements_file(judgements_file, outfile):
   records = []
   with open(judgements_file, 'r') as f:
      for line in f:
         records.append(json.loads(line.strip()))
   judgements_df = pd.DataFrame(records)
   judgements_df.columns = ["ID", "URL", "Title", "Context", "Index", "Judgement"]

   judgements_df.to_csv(outfile, index=False)


if __name__ == '__main__':
   args = get_args()
   format_judgements_file(args.judgements_file, args.outfile)
   
