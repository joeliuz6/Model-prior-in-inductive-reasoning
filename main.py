import json
import argparse

from openai import OpenAI
from hypo_generator import HypoGenerator
from hypo_evaluator import HypoEvaluator
from utils_llm import string_to_list

def load_data(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: The file {filepath} is not a valid JSON.")
        exit()

def save_rules(rules, filepath='rules.json'):
    with open(filepath, 'w', encoding='utf8') as file:
        json.dump(rules, file, ensure_ascii=False, indent=4)

def propose_rules(args):

    if args.dataset_name == 'hallucination':
        root_path = 'data/hallucination/'
        input_type = 'image_text'
    elif args.dataset_name == 'unhealthy_conversation':
        root_path = 'data/health_conversation/'
        input_type = 'text'
    elif args.dataset_name == 'funny_reddit':
        root_path = 'data/reddit_rumor/'
        input_type = 'text'
    elif args.dataset_name == 'hotel_review':
        root_path = 'data/hotel_review/'
        input_type = 'text'
    elif args.dataset_name == 'mnist':
        root_path = 'data/pneumoniamnist/'
        input_type = 'image'
    else:
        print("Dataset not supported yet")
        return None

    train_data = load_data(root_path + 'train.json')



    generator = HypoGenerator(args)
    
    if args.with_data == 'few_shot':
        if args.data_label == 'only_positive':
            prompt_file = root_path + 'few_shot/only_positive.txt'
        elif args.data_label == 'only_negative':
            prompt_file = root_path + 'few_shot/only_negative.txt'
        else:
            prompt_file = root_path + 'few_shot/with_knowledge.txt'
    elif args.with_data == 'no_data':
        prompt_file = root_path + 'zero_shot/with_knowledge.txt'
    else:
        print("Data type not supported yet")
        return None
        
    with open(prompt_file, 'r') as file:
        prompt_template = file.read()

    output_hypothesis = generator.one_time_inference_classification(prompt_template, train_data,  args.num_hypotheses, input_type,with_data=args.with_data, data_label=args.data_label)
    
    print("Proposed rules: ", output_hypothesis)

    save_rules(output_hypothesis,root_path + 'experiment_rules/' + args.with_data + "_"  + args.data_label + "_" + str(args.number_per_class) + str(args.seed) + '_rules_io.json')

    return output_hypothesis

def evaluate_rules(args):

    if args.dataset_name == 'hallucination':
        root_path = 'data/hallucination/'
        input_type = 'image_text'
    elif args.dataset_name == 'unhealthy_conversation':
        root_path = 'data/health_conversation/'
        input_type = 'text'
    elif args.dataset_name == 'funny_reddit':
        root_path = 'data/reddit_rumor/'
        input_type = 'text'
    elif args.dataset_name == 'hotel_review':
        root_path = 'data/hotel_review/'
        input_type = 'text'
    elif args.dataset_name == 'mnist':
        root_path = 'data/pneumoniamnist/'
        input_type = 'image'
    else:
        print("Dataset not supported yet")
        return None
    
    test_data = load_data(root_path + 'test.json')
    if args.evaluate_with_know == 'yes':
        evaluate_prompt = root_path + 'evaluation_prompt.txt'
    elif args.evaluate_with_know == 'no':
        evaluate_prompt = root_path + 'evaluation_prompt_no_knowledge.txt'
    else:
        print("Evaluation type not supported yet")
        return None
    
    if args.evaluate_with_know == 'yes':
        output_hypothesis = load_data(root_path + 'experiment_rules/' + args.with_data + "_"  + args.data_label + "_" + str(args.number_per_class)  +  str(args.seed) + '_rules_io.json')
    elif args.evaluate_with_know == 'no':
        output_hypothesis = load_data(root_path + args.with_data + '_'+ args.data_label + '_rules_noknow.json')
    else:
        print("Evaluation type not supported yet")
        return None

    #output_hypothesis = load_data(root_path + 'ablation_mislead.json')
    
    rules_list = string_to_list(output_hypothesis)

    metrics_list = []  
    evaluator = HypoEvaluator(args) 

    
    with open(evaluate_prompt, 'r') as file:
        eval_prompt = file.read()


    for i, rule in enumerate(rules_list):
            
            print(f"Rule {i+1}: {rule}")
            #print("Evaluating the rule...")
            metrics,_,_ = evaluator.evaluate_with_llm_selection_batchapi(eval_prompt, rule,input_type, test_data)
            metrics_list.append(
                {
                    "rule": rule,
                    "metrics": metrics
                }
            )
            print(f"Metrics: {metrics}")


    #save_rules(metrics_list,root_path + 'experiment_rules/' + args.with_data + "_"  + args.data_label + "_" + str(args.number_per_class) + str(args.seed) + '_metric_io.json')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rule Proposer and Evaluator')
    parser.add_argument('--model', type=str, default="gpt-4o-2024-08-06")
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--num_hypotheses', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--number_per_class', type=int, default=15)
    parser.add_argument('--with_data', choices=['few_shot','no_data'], default='few_shot')
    parser.add_argument('--data_label', choices=['correct','flipped','random'], default='correct')
    parser.add_argument('--evaluate_with_know', choices=['yes','no'], default='yes')
    parser.add_argument('--dataset_name', choices=['hallucination','unhealthy_conversation','funny_reddit','hotel_review','mnist'], default='unhealthy_conversation')
    args = parser.parse_args()
    client = OpenAI(api_key="") # your api key
    propose_rules(args)
    evaluate_rules(args)

    