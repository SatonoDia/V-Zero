"""
Preprocess dataset for questioner training
This script converts data to parquet format for training a questioner to generate
geometry questions from images.
"""

import argparse
import os
import json

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="data/geo/data_questioner_train", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--filter_json", default=None, help="Path to JSON file containing 'image_idx' and 'acc' to filter training data."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    filter_json_path = args.filter_json

    data_source = "zjuwh/self_train_set"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if filter_json_path is not None:
        with open(filter_json_path, "r") as f:
            filter_data = json.load(f)

        selected_indices = {
            item["image_idx"] for item in filter_data
            if 0.1 <= item.get("acc", -1) <= 0.9
        }
        selected_indices = sorted(selected_indices)

        if selected_indices:
            train_dataset = train_dataset.select(selected_indices)
        else:
            print("Warning: No samples met the acc [0.1, 0.9] criterion. Train set is empty.")

    # questioner_instruction = """<image>\nCreate an AMC 12 level multiple-choice geometry question based on the image. Let's think step by step.
    #     First, you must fully perceive the image, extracting any valuable visual information from it (including the sizes of labeled angles, lengths of line segments, and various positional relationships), 
    #     and generate a detailed visual description of the image.

    #     Then, write a mathematically rigorous multiple-choice question that includes ALL necessary conditions. Use phrases like "Given that..." or "If..." to state condition shown in visual description.
    #     The question must include four options, one of which is the correct answer. Provide the correct answer to the generated question. It must be one of A/B/C/D, and MUST BE enclosed within <answer> </answer> tags.
    #     Any question type other than multiple-choice is FORBIDDEN. 

    #     Your MUST response in this format:

    #     <description>
    #     [Visual description you extract from the image]
    #     </description>

    #     <question>
    #     [Write a complete multiple-choice question that states all necessary conditions clearly, followed by exactly 4 answer options A B C D]
    #     </question>

    #     <answer>
    #     [Give the correct answer, only the answer option(A B C D)]
    #     </answer>

    #     <confidence>
    #     [A float number between 0 and 1 indicating your confidence in the correctness of the description and question]
    #     </confidence>

    #     DO NOT output anything else—no explanations, no extra markup.
    # """
    questioner_instruction = """<image>\nCreate a multiple-choice geometry question based on the image. Let's think step by step.
        First, you must fully perceive the image, extracting any valuable visual information from it and generate a detailed visual description of the image.

        Then, write a multiple-choice question that includes necessary conditions. Use phrases like "Given that..." or "If..." to state condition shown in visual description if it's a geometry question. 
        The question must require analysis or reasoning.
        The question must include four options, one of which is the correct answer. Provide the correct answer to the generated question. It must be one of A/B/C/D, and MUST BE enclosed within <answer> </answer> tags.
        Any question type other than multiple-choice is STRICTLY FORBIDDEN. 

        Your MUST response in this format:

        <description>
        [Visual description you extract from the image]
        </description>

        <question>
        [Write a complete multiple-choice question that states all necessary conditions clearly, followed by exactly 4 answer options A B C D]
        </question>

        <answer>
        [Correct answer A/B/C/D] 
        </answer>

        DO NOT output anything else—no explanations, no extra markup.
    """
    def make_map_fn(split):
        def process_fn(example, idx):
            original_problem = example.pop("problem")
            original_answer = example.pop("answer")
            images = example.pop("images")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": questioner_instruction,
                    }
                ],
                "images": images,
                "ability": "question_generation",
                "reward_model": {
                    "style": "custom",
                    "ground_truth": original_answer
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "original_problem": original_problem,
                    "original_answer": original_answer,
                    "original_image": images,
                    "task_type": "questioner",
                    "need_solver_feedback": True,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8, load_from_cache_file=False)

    if len(test_dataset) > 2:
        small_test_dataset = test_dataset.select(range(2))
    else:
        small_test_dataset = test_dataset

    short_test_prompt = "<image>\n\nMinimal prompt."

    def make_dummy_test_fn(example, idx):
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": short_test_prompt}],
            "images": example["images"],
            "ability": "question_generation",
            "reward_model": {"style": "custom", "ground_truth": ""},
            "extra_info": {
                "split": "test",
                "index": idx,
                "original_problem": "",
                "original_answer": "",
                "task_type": "questioner",
                "need_solver_feedback": False,
            },
        }

    test_dataset = small_test_dataset.map(function=make_dummy_test_fn, with_indices=True, num_proc=2)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))