

import argparse
import os
from typing import Dict, List, Tuple

import torch

from attention_aggregation import EMBEDDING_DIM
from encoders import CodeEncoder, load_graphcodebert, load_unixcoder
from evaluate import evaluate, print_metrics
from extract_ir_features import extract_ir_features
from file_utils import load_bug_reports_robust, load_source_files_robust, save_data_robust
from finetune_contrastive import NUM_EPOCHS, build_positive_pools, contrastive_finetune
from fold_split import consecutive_folds_split
from mine_hard_negatives import mine_hard_negatives
from preprocess import GroundTruthMatcher, extract_functions_for_files
from train_ltr import DuaLocRanker, score_candidates, train_ltr


def build_embedding_caches(
    unixcoder: CodeEncoder,
    graphcodebert: CodeEncoder,
    bug_reports: Dict,
    functions_by_file: Dict[str, List[Dict[str, str]]],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    report_component_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for bug_id, data in bug_reports.items():
            report = data.get('bug_report', {})
            summary = report.get('summary', '') or ''
            description = report.get('description', '') or ''
            report_component_embeddings[bug_id] = {
                'unixcoder_summary': unixcoder.encode_one(summary),
                'unixcoder_description': unixcoder.encode_one(description),
                'graphcodebert_summary': graphcodebert.encode_one(summary),
                'graphcodebert_description': graphcodebert.encode_one(description),
            }

    function_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for file_path, funcs in functions_by_file.items():
            bodies = [f['body'] for f in funcs if f.get('body')]
            function_embeddings[file_path] = {
                'unixcoder': unixcoder.encode(bodies),
                'graphcodebert': graphcodebert.encode(bodies),
            }

    return report_component_embeddings, function_embeddings


def main():
    parser = argparse.ArgumentParser(description="Run the full DuaLoc pipeline end-to-end for one project.")
    parser.add_argument('--project', required=True,
                         help="Project key: eclipse_platform_ui, jdt, birt, swt, tomcat, or aspectj")
    parser.add_argument('--bug_reports', required=True, help='Path to <project>_bugs.json')
    parser.add_argument('--source_files', required=True, help='Path to the project source directory')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--contrastive_epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Stage 1: Preprocessing ({args.project}) ===")
    bug_reports = load_bug_reports_robust(args.bug_reports)
    source_files = load_source_files_robust(args.source_files)
    matcher = GroundTruthMatcher(list(source_files.keys()))
    ground_truth = matcher.resolve_bug_reports(bug_reports)
    functions_by_file = extract_functions_for_files(source_files)
    print(f"  {len(bug_reports)} bug reports, {len(source_files)} source files, "
          f"{sum(len(v) for v in functions_by_file.values())} functions extracted.")

    print("\n=== Stage 2: Hard-negative mining (phi1, top-200 per report) ===")
    from tfidf_utils import build_project_tfidf_index
    tfidf_index = build_project_tfidf_index(source_files, functions_by_file, bug_reports)
    hard_negatives = mine_hard_negatives(tfidf_index, bug_reports, ground_truth)
    save_data_robust(hard_negatives, os.path.join(args.output_dir, 'hard_negatives.json'))

    print("\n=== Stage 3: IR features (phi1-phi4) ===")
    candidates = {
        bug_id: sorted(set(ground_truth.get(bug_id, [])) | set(hard_negatives.get(bug_id, [])))
        for bug_id in bug_reports
    }
    raw_ir_features = extract_ir_features(bug_reports, candidates, ground_truth, tfidf_index)
    raw_ir_features_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_ir_features.items()}

    print("\n=== Stage 4: Contrastive fine-tuning (independent InfoNCE per encoder) ===")
    fixing_files_map = {bug_id: set(files) for bug_id, files in ground_truth.items() if files}
    positive_pools = build_positive_pools(bug_reports, ground_truth, source_files)
    unixcoder = load_unixcoder(device=args.device)
    graphcodebert = load_graphcodebert(device=args.device)
    contrastive_finetune(unixcoder, bug_reports, positive_pools, fixing_files_map, num_epochs=args.contrastive_epochs)
    contrastive_finetune(graphcodebert, bug_reports, positive_pools, fixing_files_map, num_epochs=args.contrastive_epochs)
    unixcoder.save(os.path.join(args.output_dir, 'unixcoder_finetuned.pt'))
    graphcodebert.save(os.path.join(args.output_dir, 'graphcodebert_finetuned.pt'))

    print("\n=== Precomputing embedding caches (encoders now frozen) ===")
    report_component_embeddings, function_embeddings = build_embedding_caches(
        unixcoder, graphcodebert, bug_reports, functions_by_file,
    )

    print("\n=== Stages 5-6: Attention aggregation + LTR training, per consecutive fold ===")
    fold_pairs = consecutive_folds_split(bug_reports, args.project)
    all_metrics = []
    for i, (train_reports, test_reports) in enumerate(fold_pairs, start=1):
        print(f"\n--- Fold pair {i}/{len(fold_pairs)}: train {len(train_reports)} reports, "
              f"test {len(test_reports)} reports ---")

        train_pairs = []
        for bug_id in train_reports:
            fixed = set(ground_truth.get(bug_id, []))
            for fp in candidates.get(bug_id, []):
                train_pairs.append((bug_id, fp, fp in fixed))

        ranker = DuaLocRanker(d=EMBEDDING_DIM).to(args.device)
        ranker, stats = train_ltr(
            ranker, train_pairs, report_component_embeddings, function_embeddings, raw_ir_features_t,
        )

        rankings = {}
        for bug_id in test_reports:
            cand_files = candidates.get(bug_id, [])
            if not cand_files:
                continue
            rankings[bug_id] = score_candidates(
                ranker, stats, bug_id, cand_files,
                report_component_embeddings[bug_id], function_embeddings, raw_ir_features_t,
            )

        fold_ground_truth = {bug_id: ground_truth.get(bug_id, []) for bug_id in test_reports}
        metrics = evaluate(rankings, fold_ground_truth)
        print_metrics(metrics)
        all_metrics.append(metrics)

        torch.save(ranker.state_dict(), os.path.join(args.output_dir, f'ranker_fold_{i}.pt'))

    print("\n=== Stage 7: Averaged metrics across fold pairs ===")
    if all_metrics:
        avg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        print_metrics(avg)
        save_data_robust(avg, os.path.join(args.output_dir, 'final_metrics.json'))


if __name__ == '__main__':
    main()
