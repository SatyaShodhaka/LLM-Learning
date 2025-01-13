import openai
import asyncio
from typing import Any
import argparse
from datasets import load_dataset
import os
from tqdm import tqdm
import time

openai.api_key = "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"
async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def generate_predictions(prompts,args):
    predictions = asyncio.run(
        dispatch_openai_requests(
            messages_list=prompts,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
    )
    outputs = []
    questions = []
    answers = []
    for i, x in enumerate(predictions):
        output = x['choices'][0]['message']['content']
        outputs.append(output)
        question_answer_list = output.split('\n')
        if 'Question:' in question_answer_list[0]:
            question = question_answer_list[0].replace('Question: ','')
            questions.append(question)
        else:
            question.append('None')
        if 'Answer:' in question_answer_list[1]:
            answer = question_answer_list[1].replace('Answer: ','')
            answers.append(answer)
        else:
            answers.append('None')
    return outputs, questions, answers
    #return predictions

def create_prompts(dataset):
    template = """You are a knowledgeable question generator tasked with creating one clear and concise question based on a given context. Ensure that the generated question can be comprehended and answered independently, solely by examining the question itself. Additionally, provide a brief answer that is firmly rooted in the provided context. 
    
    Context: The president of the United States (POTUS) is the head of state and head of government of the United States. The president directs the executive branch of the federal government and is the commander-in-chief of the United States Armed Forces. 
    Question: What position does the president of the Untied States hold in the United States Armed Forces?
    Answer: commander-in-chief

    Context: Turkey played a prominent role in the Korean War and joined NATO in 1952. During the Cold War years, the country endured two military coups in 1960 and 1980, and a period of economic and political turmoil in the 1970s. 
    Question: When did Turkey join NATO?
    Answer: 1952

    Context: The chancellor of Germany, officially the federal chancellor of the Federal Republic of Germany, is the head of the federal government of Germany, and the commander in chief of the German Armed Forces during wartime. The chancellor is the chief executive of the Federal Cabinet and heads the executive branch. The chancellor is elected by the Bundestag on the proposal of the federal president and without debate (Article 63 of the German Constitution). The current officeholder is Olaf Scholz of the SPD, who was elected in December 2021, succeeding Angela Merkel. He was elected after the SPD entered into a coalition agreement with Alliance 90/The Greens and the FDP. 
    Question: Who succeeded Angela Merkel as the chancellor of Germany?
    Answer: Olaf Scholz

    Context: Sputnik 1 was the first artificial Earth satellite. It was launched into an elliptical low Earth orbit by the Soviet Union on 4 October 1957 as part of the Soviet space program. It sent a radio signal back to Earth for three weeks before its three silver-zinc batteries ran out. Aerodynamic drag caused it to fall back into the atmosphere on 4 January 1958. 
    Question: When was Sputnik 1 launched into space?
    Answer: 4 October 1957

    Context: Whitney Houston made her acting debut with the romantic thriller film The Bodyguard (1992), which despite its mixed reviews became the tenth highest-grossing film to that date. Its soundtrack won the Grammy Award for Album of the Year and remains the bestselling soundtrack album of all time. It generated multiple hit singles, including "I Have Nothing", "I'm Every Woman" and "I Will Always Love You"; the latter won the Grammy Award for Record of the Year, spent a then-record 14 weeks atop the Billboard Hot 100 and became the best-selling physical single by a woman in music history. 
    Question: What genre was the film in which Whitney Houston made her acting debut?
    Answer: romantic thriller

    Context: {context}
    """
    prompts = []
    for data in dataset:
        formatted_input_dict = {}
        formatted_input_dict['role'] = 'user'
        formatted_input_dict['content'] = template.format(context=data['input'])
        prompts.append([formatted_input_dict])
    return prompts

def main(args):
    #dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.dataset_name == 'WikiMIA':
        dataset = load_dataset("swj0419/WikiMIA", split="WikiMIA_length32")
    
    prompts = create_prompts(dataset)
    questions = []
    answers = []
    predictions =[]
    # for i in tqdm(range(0,len(prompts),50)):
    #     prediction, question, answer = generate_predictions(prompts[i:i+50],args)
    #     questions.extend(question)
    #     answers.extend(answer)
    #     predictions.extend(prediction)
    #     time.sleep(60)

    none_predictions = 0
    for i in tqdm(prompts):
        max_retries = 5
        retry_count = 0
        response = None
        while retry_count < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=i,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                )
                break
            except Exception as e:
                print(f"Request failed with error: {e}")
                retry_count += 1
                time.sleep(2 ** retry_count) 
        if response is None:
            print(f"Request failed after maximum retries for {i}")
            none_predictions += 1
        else:
            
            output = response['choices'][0]['message']['content']
            predictions.append(output)
            question_index = output.find('Question:')
            answer_index = output.find('Answer:')

            if question_index == -1 or answer_index == -1:
                none_predictions += 1
                questions.append('None')
                answers.append('None')
                continue

            question = output[question_index:answer_index]
            question = question.replace('\n',' ')
            question = question.strip()
            question = question.replace('Question: ','')
            questions.append(question)

            answer = output[answer_index:]
            answer = answer.replace('Answer: ','')
            answers.append(answer)

    print(f'None predictions: {none_predictions}')

    dataset = dataset.add_column('id',list(range(len(dataset))))
    dataset = dataset.add_column('question',questions)
    dataset = dataset.add_column('answer',answers)
    dataset = dataset.add_column('prediction',predictions)
    file_name = args.data_path.split('/')[-1].split('.')[0]
    new_file_name = f'{file_name}.jsonl'
    dataset.to_json(os.path.join('../data/chat_gpt_questions',new_file_name))

    #create a tsv with id, question 
    with open(os.path.join('../data/chat_gpt_questions',f'{file_name}.tsv'),'w') as f:
        for i, question in enumerate(questions):
            f.write(f'{i}\t{question}\n')
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--data_path", type=str, default="../data/chat_gpt_questions/wikimia.jsonl")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dataset_name", type=str, default="WikiMIA")
    args = parser.parse_args()
    main(args)