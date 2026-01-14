import argparse
import base64
import json
import os
import threading
import re
from io import BytesIO

from flask import Flask, request, jsonify
import vllm
from PIL import Image
from mathruler.grader import extract_boxed_content, grade_answer

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
parser.add_argument('--gpu_mem_util', type=float, default=0.9)
parser.add_argument('--tensor_parallel_size', type=int, default=1)
args = parser.parse_args()

print('[init] loading model …')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    max_model_len=8192,
    tensor_parallel_size=args.tensor_parallel_size
)

sample_params = vllm.SamplingParams(
    max_tokens=2048,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10,
)

SOLVER_SYSTEM_PROMPT = "You are an expert competition-math problem solver."

ANSWER_INSTRUCTION = r"""
    Solve the multiple-choice question based on the provided image. Let's think step by step.
    First, you must fully perceive the image, extracting any valuable visual information from it.
    Then, solve the question, give the correct choice option.

    The reasoning process MUST BE enclosed within <think> </think> tags.
    The final answer MUST BE single option A/B/C/D put in \boxed{}.
    """


def load_image_from_payload(payload):
    """
    Load a PIL Image from various possible payload types:
    """
    if payload is None:
        return None

    try:
        image_bytes = None

        if isinstance(payload, dict):
            image_bytes = payload.get("bytes")
            image_bytes = bytes(image_bytes, encoding='utf8')
        elif isinstance(payload, str):
            image_bytes = base64.b64decode(payload)

        if image_bytes:
            with Image.open(BytesIO(image_bytes)) as img:
                return img.convert("RGB")
    except Exception as e:
        print(f"[server] failed to load image payload: {e}")

    return None


class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout_sec: int, *args, **kwargs):
    result_holder = {}
    error_holder = {}

    def _target():
        try:
            result_holder['value'] = func(*args, **kwargs)
        except Exception as e:
            error_holder['error'] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_sec)

    if t.is_alive():
        raise TimeoutException()
    if 'error' in error_holder:
        raise error_holder['error']
    return result_holder.get('value')


def extract_conf(text: str) -> int:
    stripped = text.strip()
    match = re.search(r"<confidence>(.*?)</confidence>", stripped, re.IGNORECASE | re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if candidate == '0':
            return 0
        elif candidate == '1':
            return 1
    # fallback: tag missing, not '0'/'1', or invalid → return 1
    return 1


app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    name = request.args.get('name', 'None')
    print(f'[server] received {name}')

    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    prepared_inputs = []
    mapping = []
    prepared_meta = []
    results_all = [None] * len(data)

    for idx, item in enumerate(data):
        question = item.get('question', '')
        golden_answer = item.get('answer', '')
        image_payload = item.get('image','')

        if question and golden_answer and image_payload:
            image = load_image_from_payload(image_payload)
            if image is not None:
                messages = [
                    {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"{ANSWER_INSTRUCTION}\nQuestion: {question}"},
                        ],
                    },
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=True,
                )
                prepared_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            else:
                print(f'[server] Lack of image for question')
                chat = [
                    {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"{ANSWER_INSTRUCTION}\nQuestion: {question}"},
                ]
                if tokenizer.chat_template:
                    prompt = tokenizer.apply_chat_template(
                        chat,
                        tokenize=False,
                        add_generation_prompt=True,
                        add_special_tokens=True,
                    )
                else:
                    prompt = 'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']
                prepared_inputs.append({"prompt": prompt})

            mapping.append(idx)
            prepared_meta.append({"question": question, "answer": golden_answer})
        else:
            results_all[idx] = {'question': question, 'answer': golden_answer, 'score': -1, 'results': []}

    print(f'[server] prepared {len(prepared_inputs)} inputs for generation.')

    if prepared_inputs:
        responses = model.generate(prepared_inputs, sampling_params=sample_params, use_tqdm=True)
    else:
        responses = []
    print('[server] generation completed.')

    def process_single(question, golden_answer, response):
        # for out in response.outputs:
        #     conf = extract_conf(out.text)
        #     if conf == 0:
        #         print('[server] detected unsolvable or incorrect question.')
        #         return {
        #             'question': question,
        #             'answer': '',
        #             'score': 0.0,
        #             'results': []
        #         }
        results = [extract_boxed_content(out.text) for out in response.outputs]

        answer_counts = {}
        for res in results:
            matched = False
            for exist_ans in list(answer_counts.keys()):
                try:
                    if res == exist_ans:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break
                except Exception:
                    continue
            if not matched and res:
                answer_counts[res] = 1

        max_count = max(answer_counts.values()) if answer_counts else 0
        majority_ans = max(answer_counts, key=answer_counts.get) if answer_counts else ''
        score = max_count / len(results) if results else 0.0
        golden_answer_match = re.search(r'[A-Z]', golden_answer)
        if golden_answer_match:
            golden_answer_extract = golden_answer_match.group(0)
        else:
            golden_answer_extract = ''
            golden_answer = ''

        if golden_answer == '':
            score = -1
        elif grade_answer(majority_ans, golden_answer_extract):
            score = min(score, 1-score)
        elif majority_ans != 'None':
            score = 0.5 * score
        else:
            score = 0.0

        return {
            'question': question,
            'answer': majority_ans,
            'score': score,
            'results': results
        }

    response_idx = 0
    for meta, mapped_idx in zip(prepared_meta, mapping):
        try:
            response = responses[response_idx]
            response_idx += 1
            item = run_with_timeout(process_single, 10, meta["question"], meta["answer"], response)
            results_all[mapped_idx] = item
        except TimeoutException:
            print(f'[server] timeout: {meta["question"]}')
            print(f'[server] timeout: {meta["answer"]}')
            results_all[mapped_idx] = {
                'question': meta["question"],
                'answer': meta["answer"],
                'score': -1,
                'results': [],
                'error': 'timeout'
            }
        except Exception as e:
            print(f'[server] error: {e}')
            results_all[mapped_idx] = {'question': meta["question"], 'answer': meta["answer"], 'score': -1, 'results': []}

    for idx, res in enumerate(results_all):
        if res is None:
            question = data[idx].get('question', '')
            answer = data[idx].get('answer', '')
            results_all[idx] = {'question': question, 'answer': answer, 'score': -1, 'results': []}

    print('[server] results_all completed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    print(f'[server] processed {name}, results saved to {out_path}.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(args.port), threaded=True)