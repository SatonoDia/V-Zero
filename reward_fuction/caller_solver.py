import re

from mathruler.grader import extract_boxed_content, grade_answer


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info):
    return acc_reward(solution_str, ground_truth, True)