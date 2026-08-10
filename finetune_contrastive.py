

import argparse
import random
from typing import Dict, List, Set

import torch
import torch.nn.functional as F

from encoders import CodeEncoder, load_graphcodebert, load_unixcoder
from file_utils import load_bug_reports_robust, load_source_files_robust
from preprocess import GroundTruthMatcher, extract_functions

BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.07
NUM_EPOCHS = 5


def build_positive_pools(
    bug_reports: Dict,
    ground_truth: Dict[str, List[str]],
    source_files: Dict[str, str],
) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    function_cache: Dict[str, List[Dict[str, str]]] = {}

    for bug_id, fixing_files in ground_truth.items():
        pool: List[str] = []
        for fp in fixing_files:
            if fp not in source_files:
                continue
            if fp not in function_cache:
                function_cache[fp] = extract_functions(source_files[fp])
            pool.extend(f['body'] for f in function_cache[fp] if f.get('body'))
        if pool:
            pools[bug_id] = pool

    return pools


def build_query_texts(bug_reports: Dict, report_ids: List[str]) -> Dict[str, str]:

    texts = {}
    for bug_id in report_ids:
        report = bug_reports[bug_id].get('bug_report', {})
        summary = report.get('summary', '') or ''
        description = report.get('description', '') or ''
        texts[bug_id] = f"{summary} {description}".strip()
    return texts


def build_batches(
    report_ids: List[str],
    fixing_files_map: Dict[str, Set[str]],
    batch_size: int,
    rng: random.Random,
) -> List[List[str]]:

    remaining = list(report_ids)
    rng.shuffle(remaining)
    batches: List[List[str]] = []

    while remaining:
        batch: List[str] = []
        used_files: Set[str] = set()
        leftover: List[str] = []

        for rid in remaining:
            files = fixing_files_map.get(rid, set())
            if len(batch) < batch_size and not (files & used_files):
                batch.append(rid)
                used_files |= files
            else:
                leftover.append(rid)

        if batch:
            batches.append(batch)
        else:

            batches.extend([[rid] for rid in leftover])
            leftover = []

        remaining = leftover

    return batches


def contrastive_finetune(
    encoder: CodeEncoder,
    bug_reports: Dict,
    positive_pools: Dict[str, List[str]],
    fixing_files_map: Dict[str, Set[str]],
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    temperature: float = TEMPERATURE,
    max_nl_length: int = 512,
    max_pl_length: int = 512,
    seed: int = 42,
) -> CodeEncoder:
    report_ids = list(positive_pools.keys())
    query_texts = build_query_texts(bug_reports, report_ids)

    encoder.unfreeze()
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = random.Random(seed)

    for epoch in range(num_epochs):
        epoch_positive = {rid: rng.choice(positive_pools[rid]) for rid in report_ids}

        batches = build_batches(report_ids, fixing_files_map, batch_size, rng)
        rng.shuffle(batches)

        epoch_loss, num_batches = 0.0, 0
        for batch_ids in batches:
            if len(batch_ids) < 2:
                continue

            queries = [query_texts[rid] for rid in batch_ids]
            positives = [epoch_positive[rid] for rid in batch_ids]

            report_embs = encoder.encode(queries, max_length=max_nl_length)
            func_embs = encoder.encode(positives, max_length=max_pl_length)

            report_embs = F.normalize(report_embs, p=2, dim=1)
            func_embs = F.normalize(func_embs, p=2, dim=1)

            sim_matrix = (report_embs @ func_embs.T) / temperature
            labels = torch.arange(len(batch_ids), device=sim_matrix.device)
            loss = F.cross_entropy(sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"  [{encoder.model_name}] epoch {epoch + 1}/{num_epochs}: "
              f"{num_batches} batches, avg InfoNCE loss = {avg_loss:.4f}")

    encoder.freeze()
    return encoder


def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tune UniXcoder and GraphCodeBERT independently.")
    parser.add_argument('--bug_reports', required=True)
    parser.add_argument('--source_files', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    bug_reports = load_bug_reports_robust(args.bug_reports)
    source_files = load_source_files_robust(args.source_files)

    matcher = GroundTruthMatcher(list(source_files.keys()))
    ground_truth = matcher.resolve_bug_reports(bug_reports)
    fixing_files_map = {bug_id: set(files) for bug_id, files in ground_truth.items() if files}

    print("Building function-level positive pools ...")
    positive_pools = build_positive_pools(bug_reports, ground_truth, source_files)
    print(f"  {len(positive_pools)}/{len(bug_reports)} reports have >=1 usable weak positive.")

    for name, loader in (('unixcoder', load_unixcoder), ('graphcodebert', load_graphcodebert)):
        print(f"\nFine-tuning {name} independently ({args.epochs} epochs, batch size {args.batch_size}) ...")
        encoder = loader(device=args.device)
        contrastive_finetune(
            encoder, bug_reports, positive_pools, fixing_files_map,
            num_epochs=args.epochs, batch_size=args.batch_size,
        )
        save_path = os.path.join(args.output_dir, f"{name}_finetuned.pt")
        encoder.save(save_path)
        print(f"  Saved fine-tuned {name} to {save_path}")


if __name__ == '__main__':
    main()
