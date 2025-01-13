import argparse
import json
import pandas as pd

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--wikimia_raw", required=True, type=str) # path to raw WikiMIA dataset
   parser.add_argument("--raw_output", required=True, type=str) # wikimia raw to csv
   parser.add_argument("--openmatch_output", required=True, type=str) # path to the openmatch-formatted output
   parser.add_argument("--corpus_output", required=True, type=str) # path to the corpus-formatted output
   args = parser.parse_args()
   return args


def format_wikimia(wikimia, raw_output, openmatch, corpus):
   records = []
   with open(wikimia, 'r') as f:
      for line in f:
         records.append(json.loads(line.strip()))
   
   df = pd.DataFrame(records)
   df.columns = ['Context', 'gold_label']
   # df = df[df['gold_label'] == 0]
   df.index.name = 'id'
   df.to_csv(raw_output)

   openmatch_df = df[['Context']]
   openmatch_df.columns = ['context']
   openmatch_df.index.name = 'id'
   openmatch_df.to_csv(openmatch, sep="\t", header=False)

   corpus_df = openmatch_df
   corpus_df.columns = ['context']
   corpus_df.index.name = 'id'
   corpus_df['title'] = ""
   corpus_df = corpus_df[['title', 'context']]
   corpus_df.to_csv(corpus, sep="\t", header=False)

   

if __name__ == "__main__":
   args = get_args()
   format_wikimia(args.wikimia_raw, args.raw_output, args.openmatch_output, args.corpus_output)
   