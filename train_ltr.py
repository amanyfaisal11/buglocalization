

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_aggregation import EMBEDDING_DIM, FunctionToFileAggregator

HUBER_EPSILON = 0.1
L1_LAMBDA = 1e-4
L1_RATIO = 1.0
BATCH_SIZE = 256
MAX_EPOCHS = 50
EARLY_STOP_MIN_DELTA = 1e-3
EARLY_STOP_PATIENCE = 5

FEATURE_ORDER = ['unixcoder_summary', 'unixcoder_description',
                  'graphcodebert_summary', 'graphcodebert_description',
                  'phi1', 'phi2', 'phi3', 'phi4']


class DuaLocRanker(nn.Module):

    def __init__(self, d: int = EMBEDDING_DIM, h: int = 256):
        super().__init__()
        self.aggregator = FunctionToFileAggregator(d, h)
        self.w = nn.Linear(8, 1, bias=False)

    def raw_phi(
        self,
        report_component_embeddings: Dict[str, torch.Tensor],
        function_embeddings: Dict[str, torch.Tensor],
        ir_features: torch.Tensor,
    ) -> torch.Tensor:

        sigma_scores = self.aggregator.compute_scores(report_component_embeddings, function_embeddings)
        sigma_vec = torch.stack([sigma_scores[k] for k in self.aggregator.score_order()])
        return torch.cat([sigma_vec, ir_features])

    def score(self, phi_normalized: torch.Tensor) -> torch.Tensor:
        return self.w(phi_normalized.unsqueeze(0)).squeeze()


@dataclass
class MinMaxStats:
    min: torch.Tensor
    max: torch.Tensor

    def normalize(self, phi: torch.Tensor) -> torch.Tensor:
        span = (self.max - self.min).clamp(min=1e-12)
        normalized = (phi - self.min) / span
        return normalized.clamp(0.0, 1.0)


def ameliorated_target(phi_normalized: torch.Tensor, is_relevant: bool) -> torch.Tensor:
    total = phi_normalized.sum()
    if is_relevant:
        total = total + phi_normalized.max()
    return total


def huber_l1_loss(scores: torch.Tensor, targets: torch.Tensor, w: torch.Tensor,
                   epsilon: float = HUBER_EPSILON, lam: float = L1_LAMBDA, rho: float = L1_RATIO) -> torch.Tensor:
    huber = F.huber_loss(scores, targets, delta=epsilon, reduction='mean')
    l1_penalty = lam * rho * w.abs().sum()
    return huber + l1_penalty


def fit_minmax_stats(raw_phi_list: List[torch.Tensor]) -> MinMaxStats:
    stacked = torch.stack(raw_phi_list)
    return MinMaxStats(min=stacked.min(dim=0).values.detach(), max=stacked.max(dim=0).values.detach())


def _huber_dloss(p: float, y: float, epsilon: float) -> float:
    r = p - y
    abs_r = abs(r)
    if abs_r <= epsilon:
        return r
    return epsilon if r > 0.0 else -epsilon


class OptimalLR(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer: torch.optim.Optimizer, alpha: float = L1_LAMBDA, epsilon: float = HUBER_EPSILON):
        typw = (1.0 / (alpha ** 0.5)) ** 0.5
        initial_eta0 = typw / max(1.0, _huber_dloss(-typw, 1.0, epsilon))
        t0 = 1.0 / (initial_eta0 * alpha)
        super().__init__(optimizer, lr_lambda=lambda step: 1.0 / (alpha * (t0 + step)))


def train_ltr(
    ranker: DuaLocRanker,
    train_pairs: List[Tuple[str, str, bool]],
    report_component_embeddings: Dict[str, Dict[str, torch.Tensor]],
    function_embeddings: Dict[str, Dict[str, torch.Tensor]],
    raw_ir_features: Dict[Tuple[str, str], torch.Tensor],
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    seed: int = 42,
) -> Tuple[DuaLocRanker, MinMaxStats]:
    import random
    rng = random.Random(seed)

    def compute_raw_phi(bug_id: str, file_path: str) -> torch.Tensor:
        return ranker.raw_phi(
            report_component_embeddings[bug_id],
            function_embeddings[file_path],
            raw_ir_features[(bug_id, file_path)],
        )

    with torch.no_grad():
        raw_phis = [compute_raw_phi(bug_id, fp) for bug_id, fp, _ in train_pairs]
    stats = fit_minmax_stats(raw_phis)

    optimizer = torch.optim.SGD(ranker.parameters(), lr=1.0)
    scheduler = OptimalLR(optimizer)

    prev_loss = None
    no_improve_epochs = 0

    for epoch in range(max_epochs):
        pairs = list(train_pairs)
        rng.shuffle(pairs)

        epoch_loss_sum, n_batches = 0.0, 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            if not batch:
                continue

            scores, targets = [], []
            for bug_id, file_path, is_relevant in batch:
                raw_phi = compute_raw_phi(bug_id, file_path)
                phi_norm = stats.normalize(raw_phi)
                scores.append(ranker.score(phi_norm))
                targets.append(ameliorated_target(phi_norm, is_relevant))

            scores_t = torch.stack(scores)
            targets_t = torch.stack(targets).detach()

            loss = huber_l1_loss(scores_t, targets_t, ranker.w.weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss_sum += loss.item()
            n_batches += 1

        epoch_loss = epoch_loss_sum / max(1, n_batches)
        print(f"  epoch {epoch + 1}/{max_epochs}: train loss = {epoch_loss:.6f}")

        if prev_loss is not None and (prev_loss - epoch_loss) < EARLY_STOP_MIN_DELTA:
            no_improve_epochs += 1
        else:
            no_improve_epochs = 0
        prev_loss = epoch_loss

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping: no >{EARLY_STOP_MIN_DELTA} improvement for "
                  f"{EARLY_STOP_PATIENCE} consecutive epochs.")
            break

    return ranker, stats


def score_candidates(
    ranker: DuaLocRanker,
    stats: MinMaxStats,
    bug_id: str,
    candidate_files: List[str],
    report_component_embeddings: Dict[str, torch.Tensor],
    function_embeddings: Dict[str, Dict[str, torch.Tensor]],
    raw_ir_features: Dict[Tuple[str, str], torch.Tensor],
) -> List[Tuple[str, float]]:
    results = []
    with torch.no_grad():
        for fp in candidate_files:
            raw_phi = ranker.raw_phi(report_component_embeddings, function_embeddings[fp], raw_ir_features[(bug_id, fp)])
            phi_norm = stats.normalize(raw_phi)
            score = ranker.score(phi_norm).item()
            results.append((fp, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
