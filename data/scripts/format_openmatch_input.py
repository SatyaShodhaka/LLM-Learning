import argparse
import pandas as pd

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--wiki_file", required=True, type=str) # path to raw input file
   parser.add_argument("--outfile", required=True, type=str) # path to the output file
   args = parser.parse_args()
   return args


def format_for_openmatch(wiki_file, outfile):
   # load the data files
   wiki_df = pd.read_csv(wiki_file)

   openmatch_input = wiki_df[['ID', 'Context']]
   openmatch_input.to_csv(outfile, sep="\t",  header=False, index=False)


if __name__ == '__main__':
   args = get_args()
   format_for_openmatch(args.wiki_file, args.outfile)
   
