import re
import string
from collections import Counter

from unidecode import unidecode


def compute_f1_score(precision: float, recall: float) -> float:
    if precision + recall < 1e-15:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_qa_answer(input_string: str) -> str:
    # This is the same QA answer normalization function used in Wikontic
    # However, here we add also the removal of stopwords as originally
    # intended in the HotpotQA evaluation code
    # See https://github.com/hotpotqa/hotpot/blob/master/util.py#L237-L238
    input_string = unidecode(input_string)
    input_string = input_string.lower()

    # Remove commas and periods between digits (e.g., 7,531 or 7.531 -> 7531)
    input_string = re.sub(r"(?<=\d)[,\.](?=\d)", "", input_string)

    # Replace all other punctuation with a space
    input_string = re.sub(f"[{re.escape(string.punctuation)}]", " ", input_string)

    # Remove articles
    input_string = re.sub(r"\b(?:a|an|the)\s+", "", input_string)

    # Replace multiple spaces with a single space
    input_string = re.sub(r"\s+", " ", input_string)

    # Trim leading/trailing whitespace
    return input_string.strip()


# def normalize_qa_answer(s: str) -> str:
#     def remove_articles(text: str) -> str:
#         return re.sub(r"\b(a|an|the)\b", " ", text)

#     def white_space_fix(text: str) -> str:
#         return " ".join(text.split())

#     def remove_punc(text: str) -> str:
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text: str) -> str:
#         return text.lower()

#     s = unidecode(s)
#     return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_qa_f1_score(pred_answer: str, gt_answer: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_qa_answer(pred_answer)
    normalized_ground_truth = normalize_qa_answer(gt_answer)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0, 0.0, 0.0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = float(num_same) / len(prediction_tokens)
    recall = float(num_same) / len(ground_truth_tokens)
    f1 = compute_f1_score(precision, recall)
    return precision, recall, f1


def exact_match_score(pred_answer: str, gt_answer: str) -> int:
    return (
        1 if normalize_qa_answer(pred_answer) == normalize_qa_answer(gt_answer) else 0
    )


def evaluate_qa_answer(
    pred_answer: str, gt_answer: str, *, aliases: dict[str, list[str]] | None = None
) -> dict[str, int | float]:
    if "no," in pred_answer.lower():
        pred_answer = "no"
    elif "yes," in pred_answer.lower():
        pred_answer = "yes"
    em = exact_match_score(pred_answer, gt_answer)
    precision, recall, f1 = compute_qa_f1_score(pred_answer, gt_answer)
    if aliases is not None and pred_answer in aliases:
        for a in aliases[pred_answer]:
            alias_em = exact_match_score(a, gt_answer)
            alias_precision, alias_recall, alias_f1 = compute_qa_f1_score(a, gt_answer)
            em = max(em, alias_em)
            precision = max(precision, alias_precision)
            recall = max(recall, alias_recall)
            f1 = max(f1, alias_f1)
    metrics = {"exact_match": em, "precision": precision, "recall": recall, "f1": f1}
    return metrics
