
import argparse
import json
from typing import Dict, List

from file_utils import load_bug_reports_robust, load_source_files_robust, save_data_robust
from preprocess import GroundTruthMatcher, extract_functions_for_files
from tfidf_utils import ProjectTfidfIndex, build_project_tfidf_index

TOP_K = 200


def mine_hard_negatives(
    tfidf_index: ProjectTfidfIndex,
    bug_reports: Dict,
    ground_truth: Dict[str, List[str]],
    top_k: int = TOP_K,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for bug_id in bug_reports:
        if bug_id not in tfidf_index.report_row:
            continue
        fixed_files = set(ground_truth.get(bug_id, []))
        phi1_scores = tfidf_index.phi1_all_files(bug_id)

        candidates = [
            (fp, float(phi1_scores[i]))
            for i, fp in enumerate(tfidf_index.file_paths)
            if fp not in fixed_files
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        result[bug_id] = [fp for fp, _ in candidates[:top_k]]

    return result


def main():
    parser = argparse.ArgumentParser(description="Mine phi1-based hard negatives for a project.")
    parser.add_argument('--bug_reports', required=True, help='Path to <project>_bugs.json')
    parser.add_argument('--source_files', required=True, help='Path to the project source directory')
    parser.add_argument('--output', required=True, help='Output JSON path for hard negatives')
    parser.add_argument('--top_k', type=int, default=TOP_K)
    args = parser.parse_args()

    print(f"Loading bug reports from {args.bug_reports} ...")
    bug_reports = load_bug_reports_robust(args.bug_reports)
    print(f"Loading source files from {args.source_files} ...")
    source_files = load_source_files_robust(args.source_files)

    print("Resolving ground-truth file paths ...")
    matcher = GroundTruthMatcher(list(source_files.keys()))
    ground_truth = matcher.resolve_bug_reports(bug_reports)

    print("Extracting functions ...")
    functions_by_file = extract_functions_for_files(source_files)

    print("Building project TF-IDF index ...")
    tfidf_index = build_project_tfidf_index(source_files, functions_by_file, bug_reports)

    print(f"Mining top-{args.top_k} phi1 hard negatives per report ...")
    hard_negatives = mine_hard_negatives(tfidf_index, bug_reports, ground_truth, top_k=args.top_k)

    save_data_robust(hard_negatives, args.output, format='json')
    print(f"Saved hard negatives for {len(hard_negatives)} reports to {args.output}")


if __name__ == '__main__':
    main()
