import random
from utils_llm import pure_text_input,multi_image_input,multi_imagetext_input



class HypoGenerator:
    

    def __init__(self,args):

        self.args = args
        

    def one_time_inference_classification(self, prompt_template, train_data, num_hypotheses, input_type, with_data = 'few_shot', data_label = 'correct'):

        #print(input_type)

        random.seed(self.args.seed)


        if with_data == 'no_data':
            prompt = prompt_template.replace("{{num_hypotheses}}",str(num_hypotheses))
            print(prompt)    
            output_rules = pure_text_input(prompt,model = self.args.model)
            print(output_rules)

        elif with_data == 'few_shot':

            positive_data = [item for item in train_data if item['label'] == 'yes']
            negative_data = [item for item in train_data if item['label'] == 'no']

            sample_positive_data = random.sample(positive_data, self.args.number_per_class)
            sample_negative_data = random.sample(negative_data, self.args.number_per_class)

            if input_type == 'text':
                
                sample_pos_text  = [item['text'] for item in sample_positive_data]
                sample_neg_text  = [item['text'] for item in sample_negative_data]


                if data_label == 'correct':
                    input_text_pos =  "\n".join(f"{i + 1}. {p_tt}" for i, p_tt in enumerate(sample_pos_text))
                    input_text_neg =  "\n".join(f"{i + 1}. {n_tt}" for i, n_tt in enumerate(sample_neg_text))

                elif data_label == 'flipped':
                    input_text_pos =  "\n".join(f"{i + 1}. {n_tt}" for i, n_tt in enumerate(sample_neg_text))
                    input_text_neg =  "\n".join(f"{i + 1}. {p_tt}" for i, p_tt in enumerate(sample_pos_text))
                elif data_label == 'random':
                    sample_text = sample_pos_text + sample_neg_text
                    random.shuffle(sample_text)
                    sample_pos_text = sample_text[:self.args.number_per_class]
                    sample_neg_text = sample_text[self.args.number_per_class:]
                    input_text_pos =  "\n".join(f"{i + 1}. {p_tt}" for i, p_tt in enumerate(sample_pos_text))
                    input_text_neg =  "\n".join(f"{i + 1}. {p_tt}" for i, p_tt in enumerate(sample_neg_text))
                else:
                    print("Data label not supported yet")
                    return None

                prompt = prompt_template.replace("{{positive_examples}}",input_text_pos).replace("{{negative_examples}}",input_text_neg).replace("{{num_hypotheses}}",str(num_hypotheses))

                print(prompt)
                output_rules = pure_text_input(prompt,model = self.args.model)
                print(output_rules)
            
            elif input_type == 'image':

                sample_pos_image  = ['data/pneumoniamnist/image/' + item['image'] for item in sample_positive_data]
                sample_neg_image  = ['data/pneumoniamnist/image/' + item['image'] for item in sample_negative_data]

                if data_label == 'correct':
                    input_image_pos =  sample_pos_image
                    input_image_neg =  sample_neg_image
                elif data_label == 'flipped':
                    input_image_pos =  sample_neg_image
                    input_image_neg =  sample_pos_image
                elif data_label == 'random':
                    sample_image = sample_pos_image + sample_neg_image
                    random.shuffle(sample_image)
                    input_image_pos = sample_image[:self.args.number_per_class]
                    input_image_neg = sample_image[self.args.number_per_class:]
                else:
                    print("Data label not supported yet")
                    return None
                
                image_list = input_image_pos +  input_image_neg
                num_1 = self.args.number_per_class
                num_2 = self.args.number_per_class + 1 
                num_3 = self.args.number_per_class * 2

                prompt = prompt_template.replace("{{num_1}}",str(num_1)).replace("{{num_2}}",str(num_2)).replace("{{num_3}}",str(num_3)).replace("{{num_hypotheses}}",str(num_hypotheses))
                output_rules = multi_image_input(image_list,prompt,model = self.args.model)

                print(prompt)
                print(output_rules)

            elif input_type == 'image_text':

                sample_pos_dict = [{'text': item['text'], 'image': 'data/hallucination/image/' + item['image']} for item in sample_positive_data]
                sample_neg_dict = [{'text': item['text'], 'image': 'data/hallucination/image/' + item['image']} for item in sample_negative_data]

                if data_label == 'correct':
                    input_it_pos_dict =  sample_pos_dict
                    input_it_neg_dict =  sample_neg_dict
                elif data_label == 'flipped':
                    input_it_pos_dict =  sample_neg_dict
                    input_it_neg_dict =  sample_pos_dict
                elif data_label == 'random':
                    sample_dict = sample_pos_dict + sample_neg_dict
                    random.shuffle(sample_dict)
                    input_it_pos_dict = sample_dict[:self.args.number_per_class]
                    input_it_neg_dict = sample_dict[self.args.number_per_class:]
                else:
                    print("Data label not supported yet")
                    return None
                
                image_text_dict = input_it_pos_dict +  input_it_neg_dict
                num_1 = self.args.number_per_class
                num_2 = self.args.number_per_class + 1
                num_3 = self.args.number_per_class * 2

                prompt = prompt_template.replace("{{num_1}}",str(num_1)).replace("{{num_2}}",str(num_2)).replace("{{num_3}}",str(num_3)).replace("{{num_hypotheses}}",str(num_hypotheses))
                output_rules = multi_imagetext_input(prompt,image_text_dict,model = self.args.model)
            
                print(prompt) 
                print(output_rules)   

            else:
                print("Input data type not supported yet")
                output_rules = None
                return None
        else:
            print("Data type not supported yet")
            output_rules = None
            return None



        return output_rules
    

                
        