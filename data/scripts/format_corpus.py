import argparse
import json
import pandas as pd

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--psg_file", required=True, type=str) # path to raw input file
   parser.add_argument("--outfile", required=True, type=str) # path to the output file
   args = parser.parse_args()
   return args


def format_corpus(psg_file, outfile):
   # load the data files
   records = []
   with open(psg_file, 'r') as f:
      for line in f:
         records.append(json.loads(line.strip()))
   psg_df = pd.DataFrame(records)

   psg_df.to_csv(outfile, header = ["ID", "URL", "Title", "Context", "Index"], index=False)


if __name__ == '__main__':
   args = get_args()
   format_corpus(args.psg_file, args.outfile)
   
