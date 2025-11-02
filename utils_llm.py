from openai import OpenAI
import base64
import re


client = OpenAI(api_key="") # your api key


def string_to_list(string):


    pattern = r'\d+\.\s'
    split_text = re.split(pattern, string)[1:]
    rules_list = [item.strip() for item in split_text if item.strip()]
    print("Rules: ", rules_list)

    return rules_list


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def multi_imagetext_input(prompt,image_questions, model = "gpt-4o-2024-08-06"):
    
    content_list = [
        item
        for index, one_image in enumerate(image_questions)
        for item in (
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encode_image(one_image['image'])}"}},  ### our dataset is jpg
            {"type": "text", "text": f"Image Number: {index + 1}, Questions: {one_image['text']}"},
        )
    ]
    content_list.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content_list
            }
        ],
        temperature=1e-15,
        top_p=1e-10,
    )

    return response.choices[0].message.content


def multi_image_input(image_list,prompt,model = "gpt-4o-2024-08-06"):
    
    content_list = [
        item
        for index, one_image in enumerate(image_list)
        for item in (
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(one_image)}"}}, ### our dataset is png for image only
            {"type": "text", "text": f"Image Number: {index + 1}"},
        )
    ]
    
    content_list.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content_list
            }
        ],
        temperature=1e-15,
        top_p=1e-10,
    )

    return response.choices[0].message.content


def pure_text_input(prompt,model = "gpt-4o-2024-08-06"):
    
    

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1e-15,
        top_p=1e-10,
    )

    return response.choices[0].message.content


def get_text_output_evaluate_batchAPI(id,prompt, model = "gpt-4o-2024-08-06",max_tokens = 2048):


    one_line_input = {
        "custom_id": "request-" + str(id), 
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 1e-15,
            "top_p": 1e-10,
            "max_tokens": max_tokens
        }
    }

    return one_line_input

def get_vlm_output_evaluate_jpg_batchAPI(id,image_path, prompt, model = "gpt-4o-2024-08-06",max_tokens = 2048):

    
    base64_image = encode_image(image_path)

    one_line_input = {
        "custom_id": "request-" + str(id), 
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{base64_image}",      ### our dataset is jpg
                            },
                        },
                    ],
                }
            ],
            "temperature": 1e-15,
            "top_p": 1e-10,
            "max_tokens": max_tokens
        }
    }

    return one_line_input


def get_image_output_evaluate_png_batchAPI(id,image_path, prompt, model = "gpt-4o-2024-08-06",max_tokens = 2048):

    
    base64_image = encode_image(image_path)

    one_line_input = {
        "custom_id": "request-" + str(id), 
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",   ### our dataset is png
                            },
                        },
                    ],
                }
            ],
            "temperature": 1e-15,
            "top_p": 1e-10,
            "max_tokens": max_tokens
        }
    }

    return one_line_input