
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_top_k_accuracy(rankings: Dict[str, List[Tuple[str, float]]],
                             ground_truth: Dict[str, List[str]],
                             k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
    accuracies = {k: 0.0 for k in k_values}
    total_queries = 0

    for bug_id, ranked_files in rankings.items():
        if bug_id not in ground_truth:
            continue

        relevant_files = set(ground_truth[bug_id])
        if not relevant_files:
            continue

        total_queries += 1
        ranked_paths = [path for path, _ in ranked_files]

        for k in k_values:
            top_k_paths = set(ranked_paths[:k])
            if relevant_files & top_k_paths:
                accuracies[k] += 1.0

    for k in k_values:
        if total_queries > 0:
            accuracies[k] /= total_queries

    return accuracies


def calculate_map(rankings: Dict[str, List[Tuple[str, float]]],
                  ground_truth: Dict[str, List[str]]) -> float:
    average_precisions = []

    for bug_id, ranked_files in rankings.items():
        if bug_id not in ground_truth:
            continue

        relevant_files = set(ground_truth[bug_id])
        if not relevant_files:
            continue

        ranked_paths = [path for path, _ in ranked_files]
        num_relevant = len(relevant_files)

        precisions = []
        num_relevant_found = 0

        for rank, file_path in enumerate(ranked_paths, start=1):
            if file_path in relevant_files:
                num_relevant_found += 1
                precision = num_relevant_found / rank
                precisions.append(precision)

        if precisions:
            avg_precision = sum(precisions) / num_relevant
            average_precisions.append(avg_precision)

    map_score = np.mean(average_precisions) if average_precisions else 0.0
    return map_score


def calculate_mrr(rankings: Dict[str, List[Tuple[str, float]]],
                  ground_truth: Dict[str, List[str]]) -> float:
    reciprocal_ranks = []

    for bug_id, ranked_files in rankings.items():
        if bug_id not in ground_truth:
            continue

        relevant_files = set(ground_truth[bug_id])
        if not relevant_files:
            continue

        ranked_paths = [path for path, _ in ranked_files]

        for rank, file_path in enumerate(ranked_paths, start=1):
            if file_path in relevant_files:
                reciprocal_rank = 1.0 / rank
                reciprocal_ranks.append(reciprocal_rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    mrr_score = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    return mrr_score


def evaluate(rankings: Dict[str, List[Tuple[str, float]]],
            ground_truth: Dict[str, List[str]],
            k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    if not rankings or not ground_truth:
        out = {f'Top-{k}': 0.0 for k in k_values}
        out['MAP'] = 0.0
        out['MRR'] = 0.0
        return out

    results = {}

    top_k_acc = calculate_top_k_accuracy(rankings, ground_truth, k_values)
    for k in k_values:
        results[f'Top-{k}'] = top_k_acc[k]

    results['MAP'] = calculate_map(rankings, ground_truth)

    results['MRR'] = calculate_mrr(rankings, ground_truth)

    return results


def print_metrics(results: Dict[str, float]):
    if results is None:
        results = {}
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric:15s}: {value:.4f} ({value*100:.2f}%)")

    print("="*50 + "\n")

