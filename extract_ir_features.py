
import argparse
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from file_utils import load_bug_reports_robust, load_source_files_robust
from preprocess import GroundTruthMatcher, extract_functions_for_files, preprocess_text
from tfidf_utils import ProjectTfidfIndex, build_project_tfidf_index

FeatureKey = Tuple[str, str]


def build_fix_history(bug_reports: Dict, ground_truth: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, float, str]]]:
    history = defaultdict(list)
    for bug_id, files in ground_truth.items():
        report = bug_reports[bug_id].get('bug_report', {})
        ts = report.get('timestamp', 0) or 0
        summary = report.get('summary', '') or ''
        for fp in files:
            history[fp].append((bug_id, ts, summary))
    for fp in history:
        history[fp].sort(key=lambda x: x[1])
    return history


def before(bug_id: str, file_path: str, bug_reports: Dict,
           history: Dict[str, List[Tuple[str, float, str]]]) -> List[Tuple[str, float, str]]:
    r_ts = bug_reports[bug_id].get('bug_report', {}).get('timestamp', 0) or 0
    return [entry for entry in history.get(file_path, []) if entry[1] < r_ts]


def _month_of(timestamp: float) -> int:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).month


def compute_phi2(bug_id: str, file_path: str, bug_reports: Dict,
                  history: Dict[str, List[Tuple[str, float, str]]],
                  tfidf_index: ProjectTfidfIndex) -> float:
    prior = before(bug_id, file_path, bug_reports, history)
    if not prior:
        return 0.0

    doc_tokens = preprocess_text(" ".join(summary for _, _, summary in prior))
    if not doc_tokens:
        return 0.0

    doc_vec = tfidf_index.vectorizer.transform([doc_tokens])
    r_idx = tfidf_index.report_row.get(bug_id)
    if r_idx is None:
        return 0.0
    r_vec = tfidf_index.report_matrix[r_idx]
    return float((r_vec @ doc_vec.T)[0, 0])


def compute_phi3(bug_id: str, file_path: str, bug_reports: Dict,
                  history: Dict[str, List[Tuple[str, float, str]]]) -> float:
    prior = before(bug_id, file_path, bug_reports, history)
    if not prior:
        return 0.0

    _, last_ts, _ = prior[-1]
    r_ts = bug_reports[bug_id].get('bug_report', {}).get('timestamp', 0) or 0

    d = _month_of(r_ts) - _month_of(last_ts) + 1
    return 1.0 / d if d > 0 else 0.0


def compute_phi4(bug_id: str, file_path: str, bug_reports: Dict,
                  history: Dict[str, List[Tuple[str, float, str]]]) -> float:
    return float(len(before(bug_id, file_path, bug_reports, history)))


def extract_ir_features(
    bug_reports: Dict,
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    tfidf_index: ProjectTfidfIndex,
) -> Dict[FeatureKey, np.ndarray]:
    history = build_fix_history(bug_reports, ground_truth)
    features: Dict[FeatureKey, np.ndarray] = {}

    for bug_id, file_paths in candidates.items():
        if bug_id not in tfidf_index.report_row:
            continue
        phi1_all = tfidf_index.phi1_all_files(bug_id)
        for fp in file_paths:
            f_idx = tfidf_index.file_row.get(fp)
            phi1 = float(phi1_all[f_idx]) if f_idx is not None else 0.0
            phi2 = compute_phi2(bug_id, fp, bug_reports, history, tfidf_index)
            phi3 = compute_phi3(bug_id, fp, bug_reports, history)
            phi4 = compute_phi4(bug_id, fp, bug_reports, history)
            features[(bug_id, fp)] = np.array([phi1, phi2, phi3, phi4], dtype=np.float64)

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract phi1-phi4 IR features for a project.")
    parser.add_argument('--bug_reports', required=True)
    parser.add_argument('--source_files', required=True)
    parser.add_argument('--hard_negatives', required=True, help='Output of mine_hard_negatives.py')
    parser.add_argument('--output', required=True, help='Output pickle path')
    args = parser.parse_args()

    import json
    bug_reports = load_bug_reports_robust(args.bug_reports)
    source_files = load_source_files_robust(args.source_files)
    with open(args.hard_negatives, 'r', encoding='utf-8') as f:
        hard_negatives = json.load(f)

    matcher = GroundTruthMatcher(list(source_files.keys()))
    ground_truth = matcher.resolve_bug_reports(bug_reports)

    functions_by_file = extract_functions_for_files(source_files)
    tfidf_index = build_project_tfidf_index(source_files, functions_by_file, bug_reports)

    candidates = {
        bug_id: sorted(set(ground_truth.get(bug_id, [])) | set(hard_negatives.get(bug_id, [])))
        for bug_id in bug_reports
    }

    features = extract_ir_features(bug_reports, candidates, ground_truth, tfidf_index)

    with open(args.output, 'wb') as f:
        pickle.dump(features, f)
    print(f"Saved IR features for {len(features)} (report, file) pairs to {args.output}")


if __name__ == '__main__':
    main()
