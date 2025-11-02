from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils_llm import get_text_output_evaluate_batchAPI, get_image_output_evaluate_png_batchAPI,get_vlm_output_evaluate_jpg_batchAPI
from openai import BadRequestError
from batch_api_util import OpenAIBatchProcessor
import time
import re
import json



class HypoEvaluator:
    """
    Evaluate the rule effectiveness on the given dataset.
    """


    def __init__(self, args):
        self.args = args


    def evaluate_with_llm_selection_batchapi(self, evaluate_prompt, rules, input_type, test_data: List[Dict]):

        y_true = []
        y_pred = []

        input_json_file = 'input_batchapi/input_get_description.jsonl'
        result_json_file = 'output_batchapi/output_get_description.jsonl'

        api_key=""  # your api key
        processor = OpenAIBatchProcessor(api_key)

        error_cnt = 0

        with open(input_json_file, 'w',encoding='utf8') as f:

            if input_type == 'text':

                for i in range(len(test_data)):

                    data_text = test_data[i]['text']
                    prompt = evaluate_prompt.replace('{{patterns}}',rules).replace('{{text}}',data_text)
                    input_line = get_text_output_evaluate_batchAPI(i,prompt, model=self.args.model,max_tokens=self.args.max_tokens)

                    f.write(json.dumps(input_line)+'\n')

            elif input_type == 'image':

                for i in range(len(test_data)):

                    image_path = 'data/pneumoniamnist/image/' + test_data[i]['image']
                    prompt = evaluate_prompt.replace('{{patterns}}',rules)
                    input_line = get_image_output_evaluate_png_batchAPI(i,image_path, prompt, model=self.args.model,max_tokens=self.args.max_tokens)

                    f.write(json.dumps(input_line)+'\n')

            elif input_type == 'image_text':

                for i in range(len(test_data)):

                    image_path = 'data/hallucination/image/' + test_data[i]['image']
                    data_text = test_data[i]['text']
                    prompt = evaluate_prompt.replace('{{patterns}}',rules).replace('{{text}}',data_text)
                    input_line = get_vlm_output_evaluate_jpg_batchAPI(i,image_path, prompt, model=self.args.model,max_tokens=self.args.max_tokens)

                    f.write(json.dumps(input_line)+'\n')

            else:
                raise ValueError("Invalid input type. Only 'text', 'image' or 'image_text' is allowed.")

        #print('get input for batch api done')


    # get output data from batch api

    # Process the batch job
    
        endpoint = "/v1/chat/completions"
        completion_window = "24h"

        results = processor.process_batch(input_json_file, result_json_file, endpoint, completion_window)

        time.sleep(10)

        with open(result_json_file, 'r',encoding='utf8') as f:
            for i,line in enumerate(f):

                result = json.loads(line.strip())

                id  = result['custom_id'].split('-')[-1]

                if int(id) == i:

                    model_output = result["response"]["body"]["choices"][0]["message"]["content"]

                    #print('model_output:', model_output)

                    answer_match = re.search(r"Answer:\s*(\w+)", model_output)
                    answer = answer_match.group(1) if answer_match else None
                    # can not be none 
                    if answer is None:
                        
                        error_cnt += 1
                        print('error:', i)
                        continue

                    
                    if 'yes' in answer.lower():
                        model_output = 'yes'
                    elif 'no' in answer.lower():
                        model_output = 'no'
                    else:

                        error_cnt += 1
                        
                        # skip to next 
                        print('error:', i)
                        continue
                    
                    y_pred.append(model_output)
                    expected_output = test_data[i]['label']
                    y_true.append(expected_output)
                
                else:
                    error_cnt += 1
                    print('error:', i)

        # check whether only 'yes' or 'no' in y_pred
        for pred in y_pred:
            if pred not in ['yes', 'no']:
                raise ValueError("Invalid prediction. Only 'yes' or 'no' is allowed.")
            
        print('error_cnt:', error_cnt)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label="yes"),
            'recall': recall_score(y_true, y_pred, pos_label="yes"),
            'f1': f1_score(y_true, y_pred, pos_label="yes")
        }

        return metrics, y_pred,y_true
    