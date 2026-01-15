import json
import os
import re 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image
import argparse
from tqdm import tqdm
from pathlib import Path
import datasets
import base64

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(
    description='Generate geometry math questions for images using LLM')
parser.add_argument('--model_path',
                    default='/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct',
                    help='Path to model')
parser.add_argument('--data_source',
                    default='hiyouga/geometry3k', 
                    help='data source')
parser.add_argument('--input_file',
                    default=None,
                    help='Path to the image folder')
parser.add_argument('--output_file_path',
                    help='save path of generated questions',
                    default='/root/autodl-tmp/self_train_verl/results/example/questions.json')
parser.add_argument(
                    "--filter_json", default=None, help="Path to JSON file containing 'image_idx' and 'acc' to filter training data.")

def load_image_from_dataset_item(image_field):
    """
    Load a PIL Image from various possible dataset field types:
    - PIL.Image.Image → return as-is
    - bytes (PNG/JPEG data) → open with BytesIO
    - str (base64 string) → open with Image.open
    """
    if isinstance(image_field, Image.Image):
        return image_field
    elif isinstance(image_field, bytes):
        from io import BytesIO
        return Image.open(BytesIO(image_field))
    elif isinstance(image_field, str):
        image_bytes = base64.b64decode(image_field, validate=True)
        return Image.open(BytesIO(image_bytes))
    else:
        raise ValueError(f"Unsupported image field type: {type(image_field)}")
    

def split_output_text(output_text):
    """
    Extract the generated question from <question></question> tags.
    """
    text = output_text.strip()
    
    question_match = re.search(r'<question>(.*?)</question>', text, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()
    else:
        question = None
    
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = None
    
    description_match = re.search(r'<description>(.*?)</description>', text, re.DOTALL)
    if description_match:
        description = description_match.group(1).strip()
    else:
        description = None
    
    return question, answer, description



def process_geo3k_dataset(dataset, llm, sampling_params, tokenizer, indices=None):
    valid_data = []
    inputs = []
    system_message = "You are a professional question designer."
    user_prompt = '''Create a multiple-choice question based on the image. Let's think step by step.
            First, you must fully perceive the image, extracting any valuable visual information from it and generate a detailed visual description of the image.

            Then, write a multiple-choice question that includes necessary conditions. 
            Make sure the question provides sufficient information to be answered. Use phrases like "If..." or "Given that..." to state condition shown in visual description if it's a geometry question.
            The question must require analysis or reasoning.
            The question must include four options, one of which is the correct answer. 
            Provide the correct answer to the generated question without thinking. It must be one of A/B/C/D, and MUST BE enclosed within <answer> </answer> tags.
            Any question type other than multiple-choice is STRICTLY FORBIDDEN!

            Your MUST response in this format:

            <description>
            [Visual description you extract from the image]
            </description>

            <question>
            [Write a complete multiple-choice question that states all necessary conditions clearly, followed by exactly 4 answer options A B C D]
            </question>
    
            <answer>
            [Correct answer option A/B/C/D] 
            </answer>
            
            DO NOT output anything else—no explanations, no extra markup.
            '''
    
    # Determine what to iterate over
    if indices is not None:
        # Use specified indices; get items by indexing dataset
        iterable = [(idx, dataset[idx]) for idx in indices]
    else:
        # Fallback: full dataset enumeration
        iterable = enumerate(dataset)

    for original_idx, item in tqdm(iterable, desc="Preparing inputs"):
        try:
            image = item['images'][0]
            image = load_image_from_dataset_item(image)

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })

            valid_data.append({
                "img_data_source": args.data_source,
                "image_idx": original_idx  
            })

        except Exception as e:
            print(f"Error at original idx {original_idx}: {e}")
            continue


    if not inputs:
        print("No valid images found to process!")
        return []

    print(f"Generating questions for {len(inputs)} images...")
    outputs = llm.generate(inputs, sampling_params)
    data_res = []
    
    for idx, item in tqdm(enumerate(valid_data), total=len(valid_data), desc="Storing questions"):
        try:
            output_text = outputs[idx].outputs[0].text.strip()
            question, answer, description = split_output_text(output_text)
          
            result_item = {
                "img_data_source": item["img_data_source"],
                "image_idx": item["image_idx"],
                "description": description,
                "question": question,         # Add question attribute
                "answer": answer
            }
            
            data_res.append(result_item)
                
        except Exception as e:
            print(f"Error processing output for image {item['image_name']}: {str(e)}")
            continue
        
    return data_res


if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(
        temperature=1.0,              
        repetition_penalty=1.1,  
        max_tokens = 1024,
        stop_token_ids=[tokenizer.eos_token_id],   
    )
    # Load model and tokenizer once
    print("Loading model...")
    try:
        llm = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            tensor_parallel_size=2
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        exit(1)

    # Check if input folder exists
    if args.input_file is not None:
        dataset = datasets.load_dataset(
            args.input_file,
        )
    else:
        dataset = datasets.load_dataset(
            args.data_source,
        )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    if args.filter_json is not None:
        with open(args.filter_json, "r") as f:
            filter_data = json.load(f)

        selected_indices = {
            item["image_idx"] for item in filter_data
            if 0.1 <= item.get("acc", -1) <= 0.9
        }
        selected_indices = sorted(selected_indices)
    else:
        selected_indices = None

    # Process images and generate questions
    data_res = []
    try:

        data_res = process_geo3k_dataset(train_dataset, llm, sampling_params, tokenizer, selected_indices)

        if not data_res:
            print("No questions were generated!")
            exit(1)
            
        # Save results
        print(f"Saving {len(data_res)} generated questions to {args.output_file_path}")
        output_path = Path(args.output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_res, ensure_ascii=False, indent=4)) 
            print(f"Successfully generated {len(data_res)} geometry questions!")
                
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        exit(1)