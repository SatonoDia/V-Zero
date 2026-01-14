from typing import Dict, List
import json
import os
import time
import random
import requests
import base64
import re


STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp")


def generate_temp_filename(prefix="temp", suffix=".json"):
    temp_dir = f"{STORAGE_PATH}/temp_results"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{temp_dir}/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=1):
    return [lst]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index, i):
    """Call a local solver service to process the given file."""
    try:
        response = requests.get(f"http://127.0.0.1:{5000+index}/hello?name={i}", timeout=1200)
        if response.status_code == 200:
            print(f"[caller] Solver {index} processed {i}")
            return True
        else:
            print(f"[caller] Solver {index} failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[caller] Solver {index} error: {e}")
        return False

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
        answer = answer_match.group(1).strip().upper()
    else:
        answer = None
    
    confidence_match = re.search(r'<confidence>(.*?)</confidence>', text, re.DOTALL)
    if confidence_match:
        confidence = confidence_match.group(1).strip()
    else:
        confidence = 0.0
    
    return question, answer, confidence

def generate_results(data):

    random_name = generate_temp_filename(prefix="temp_0", suffix=".json")
    
    with open(random_name, 'w') as f:
        json.dump(data, f, indent=4)

    fetch(0, random_name)

    result_file = random_name.replace('.json', '_results.json')
    with open(result_file, 'r') as f:
        final_results = json.load(f)
    
    os.remove(result_file)
    
    return final_results

def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)
    # Ensure all input lists have the same length
    n = len(solution_strs)
    assert len(extra_infos) == n, "input_images length mismatch"

    valid_data = []
    valid_indices = []
    scores = [-1.0] * n  # Default score for all items; will update valid ones

    for i, sol_str in enumerate(solution_strs):
        try:
            question, answer, confidence = split_output_text(sol_str)
            image = extra_infos[i].get("original_image", None) if extra_infos[i] else None
            
            base64_str = base64.b64encode(image[0]["bytes"]).decode('ascii') if image else None

            if isinstance(answer, (int, float)):
                scores[i] = -1.0
                continue

            has_ABCD = all(opt in question for opt in ["A)", "B)", "C)", "D)"])

            if question and answer and image and has_ABCD:
                valid_data.append({"question": question, "answer": answer, "image": base64_str})
                valid_indices.append(i)
            else:
                # Missing question or answer or image → invalid
                scores[i] = -1.0
        except (TypeError, KeyError, IndexError, ValueError):
            # Invalid JSON or non-string input
            scores[i] = -1.0
    results = generate_results(valid_data)

    if len(results) != len(valid_data):
        print(f"[compute_score] Warning: expected {len(valid_data)} results, got {len(results)}")
        # Pad or truncate to match
        results = (results + [{"question": "", "score": -1.0}] * len(valid_data))[:len(valid_data)]

    for idx, result, data in zip(valid_indices, results, valid_data):
        if result.get("question") and "score" in result:
            solver_score = float(result["score"])
            scores[idx] = solver_score
        else:
            scores[idx] = -1.0
    return scores