import json

def read_json(filename):
	with open(filename, 'r') as file:
		data_list = json.load(file)
	print(f"Read {len(data_list)} samples")
	return data_list

def read_jsonl(file_path):
	data = []
	with open(file_path, 'r') as file:
		for line in file:
			# Parse each line as a JSON object and append it to the data list
			data.append(json.loads(line))
	return data

def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')

def convert_jsonl_to_dpr_format(input_file_path, output_file_path):
    json_objects = []
    with open(input_file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_object = json.loads(line)
            json_object["answers"] = [json_object["answer"]]
            del json_object["answer"]
            json_objects.append(json_object)
    with open(output_file_path, 'w') as output_file:
        # Write the entire list of JSON objects to the file
        json.dump(json_objects, output_file, indent=4)
        

if __name__ == "__main__":
    convert_jsonl_to_dpr_format("data/freshqa/freshqa_extracted_data.jsonl", "data/freshqa/freshqa_questions_2023_changing_onehop_truepremise.jsonl")