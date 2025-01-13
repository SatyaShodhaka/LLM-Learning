import argparse
import json
import pandas as pd

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--wikimia_questions", required=True, type=str) # path to wikimia questions jsonl
   parser.add_argument("--outfile", required=True, type=str) # path to the output file
   args = parser.parse_args()
   return args


def format_wikimia_questions(wikimia_questions, outfile):
   records = []
   with open(wikimia_questions, 'r') as f:
      for line in f:
         records.append(json.loads(line.strip()))
   df = pd.DataFrame(records)
   df = df.drop(columns=['prediction'])
   df.columns = ['Context', 'gold_label', 'id', 'question', 'answer']
   df.to_csv(outfile, index=False)

if __name__ == "__main__":
   args = get_args()
   format_wikimia_questions(args.wikimia_questions, args.outfile)
   