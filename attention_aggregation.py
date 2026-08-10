

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 768
ATTENTION_DIM = 256

INSTANCE_KEYS: List[Tuple[str, str]] = [
    ('unixcoder', 'summary'),
    ('unixcoder', 'description'),
    ('graphcodebert', 'summary'),
    ('graphcodebert', 'description'),
]


class BugReportConditionedAttention(nn.Module):

    def __init__(self, d: int = EMBEDDING_DIM, h: int = ATTENTION_DIM):
        super().__init__()
        self.d = d
        self.h = h
        self.W = nn.Linear(2 * d, h)
        self.u = nn.Linear(h, 1, bias=False)

    def forward(self, report_embedding: torch.Tensor, function_embeddings: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        k = function_embeddings.size(0)
        if k == 0:
            return torch.zeros_like(report_embedding), torch.zeros(0, device=report_embedding.device)

        query_expanded = report_embedding.unsqueeze(0).expand(k, -1)
        combined = torch.cat([query_expanded, function_embeddings], dim=1)
        e = self.u(torch.tanh(self.W(combined))).squeeze(-1)
        alpha = F.softmax(e, dim=0)
        v_s = torch.sum(alpha.unsqueeze(1) * function_embeddings, dim=0)
        return v_s, alpha


class FunctionToFileAggregator(nn.Module):


    def __init__(self, d: int = EMBEDDING_DIM, h: int = ATTENTION_DIM):
        super().__init__()
        self.attentions = nn.ModuleDict({
            f'{enc}_{comp}': BugReportConditionedAttention(d, h)
            for enc, comp in INSTANCE_KEYS
        })

    def compute_scores(
        self,
        report_component_embeddings: Dict[str, torch.Tensor],
        function_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        scores: Dict[str, torch.Tensor] = {}
        for enc, comp in INSTANCE_KEYS:
            key = f'{enc}_{comp}'
            v_r = report_component_embeddings[key]
            v_f = function_embeddings[enc]
            v_s, _ = self.attentions[key](v_r, v_f)
            if v_f.size(0) == 0:
                scores[key] = torch.tensor(0.0, device=v_r.device)
            else:
                scores[key] = F.cosine_similarity(v_r.unsqueeze(0), v_s.unsqueeze(0)).squeeze(0)
        return scores

    def score_order(self) -> List[str]:

        return [f'{enc}_{comp}' for enc, comp in INSTANCE_KEYS]
