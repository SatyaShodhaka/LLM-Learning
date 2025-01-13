import json
import pandas as pd

dpr = '/question2passage-dpr/results/WikiMIA/wikimia_questions_oldcorpus_llama7b_fewshot_nocontext_top1.jsonl'
judgements_file = '/data/wikimia/wikimia_corpus_judgements.csv'
out = '/question2passage-dpr/results/WikiMIA/wikimia_questions_oldcorpus_llama7b_fewshot_nocontext_top1_wanswers.jsonl'

corpus = pd.read_csv(judgements_file)

with open(dpr, 'r') as d, open(out, 'w') as o:
   for line in d:
      record = json.loads(line.strip())
      match = corpus[corpus['question'] == record['question']]
      record['answer'] = match['answer'].iloc[0]
      record['id'] = int(match['id'].iloc[0])
      record['gold_label'] = int(match['gold_label'].iloc[0])
      o.write(json.dumps(record) + "\n")
