import argparse
import json
import pandas as pd
from tqdm import tqdm

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--corpus", required=True, type=str) # path to corpus file results retrieved from
   parser.add_argument("--dataset", required=True, type=str) # path to dataset retrieval performed on
   parser.add_argument("--dataset_type", required=True, type=str) # type of input file: tsv, csv, jsonl
   parser.add_argument("--trec_file", required=True, type=str) # path to trec results file
   parser.add_argument("--trec_id_type", required=True, type=str) # id type is either int or str
   parser.add_argument("--outfile", required=True, type=str) # path to the output file
   args = parser.parse_args()
   return args


def load_jsonl(dataset):
   records = []
   with open(dataset, 'r') as f:
      for line in f:
         records.append(json.loads(line.strip()))
   return pd.DataFrame(records)


def load_tsv(dataset):
   psg_df = pd.read_csv(dataset, sep="\t", header=None)
   psg_df.columns = ['id', 'question', 'answers']
   return psg_df


def extract_trec(corpus, dataset, dataset_type, trec_file, trec_id_type, outfile):
   # Load dataset based on type, all should return a df
   if dataset_type == "jsonl":
      psg_df = load_jsonl(dataset)
   elif dataset_type == "tsv":
      psg_df = load_tsv(dataset)
   else:
      print("INVALID TYPE SPECIFIED")
      return

   # Load TREC file
   print("PARSING TREC RESULTS.........")
   json_records = []
   docids_to_find = set()
   with open(trec_file, 'r') as trec:
      curr_id = None
      question = None
      ctxs = []

      for line in trec:
         psg_id, _, docid, _, score, _ = line.split()
         if trec_id_type == "int":
            psg_id = int(psg_id)

         if psg_id != curr_id:
            # finished processing all for current ID, store record and move to next passage
            if curr_id != None:
               json_records.append({'id':str(psg_id), 'question':question, 'answers':answer, 'ctxs':ctxs})

            curr_id = psg_id
            question = psg_df[psg_df['id'] == psg_id]['question'].iloc[0]
            answer = psg_df[psg_df['id'] == psg_id]['answers'].iloc[0]
            ctxs = []
         
         docids_to_find.add(docid)
         ctxs.append({"id":docid, "title":None, "text":None, "score":str(score), "has_answer":True})
      json_records.append({'id':psg_id, 'question':question, 'answers':answer, 'ctxs':ctxs})

   # Search corpus for docids
   print("SEARCHING CORPUS.....")
   target_docs_dfs = []
   with pd.read_csv(corpus, sep="\t", chunksize=5000, header=None) as reader:
      for chunk in tqdm(reader):
         chunk.columns = ['id', 'title', 'context']
         target_docs_dfs.append(chunk[chunk['id'].isin(docids_to_find)])
   target_docs_df = pd.concat(target_docs_dfs, axis=0)

   # Update json_records with passages
   print("UPDATING JSON RECORDS.........")
   for record in tqdm(json_records):
      ctxs = record['ctxs']
      for ctx in ctxs:
         doc_record = target_docs_df[target_docs_df['id'] == ctx['id']]
         ctx['title'] = doc_record['title'].iloc[0]
         ctx['text'] = doc_record['context'].iloc[0]

   # Write results out to outfile
   with open(outfile, 'w') as f:
      f.write(json.dumps(json_records, indent=4))


if __name__ == '__main__':
   args = get_args()
   extract_trec(args.corpus, args.dataset, args.dataset_type, args.trec_file, args.trec_id_type, args.outfile)
   