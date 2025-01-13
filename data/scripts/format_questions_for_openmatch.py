import argparse
import pandas as pd

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--q_file", required=True, type=str) # path to raw input file
   parser.add_argument("--outfile", required=True, type=str) # path to the output file
   args = parser.parse_args()
   return args


def format_questions_for_openmatch(q_file, outfile):
   # Load the questions file
   df = pd.read_csv(q_file)

   openmatch = df[['index', 'question']]
   openmatch.to_csv(outfile, sep="\t", header=False, index=False)


if __name__ == '__main__':
   args = get_args()
   format_questions_for_openmatch(args.q_file, args.outfile)
   
