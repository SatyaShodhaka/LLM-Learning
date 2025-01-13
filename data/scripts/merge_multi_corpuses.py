import argparse
import pandas as pd
from tqdm import tqdm

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--corpus_1", required=True, type=str) # path to first corpus to merge
   parser.add_argument("--corpus_2", required=True, type=str) # path to second corpus to merge
   parser.add_argument("--outfile", required=True, type=str) # path to the output file
   args = parser.parse_args()
   return args


def merge_corpuses(corpus_1, corpus_2, outfile):
   with open(outfile, 'w') as f:
       f.write("id\ttitle\ttext\n")

   # load the data files
   for corpus in [corpus_1, corpus_2]:
      with pd.read_csv(corpus, sep="\t", chunksize=5000) as reader:
         for chunk in tqdm(reader):
            chunk.to_csv(outfile, sep="\t", mode="a", header=False, index=False)


if __name__ == '__main__':
   args = get_args()
   merge_corpuses(args.corpus_1, args.corpus_2, args.outfile)
   
