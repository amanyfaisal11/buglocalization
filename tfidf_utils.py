

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from preprocess import preprocess_text


class PaperTfidfVectorizer:


    def __init__(self):
        self.vocab_: Dict[str, int] = {}
        self.idf_: np.ndarray = None
        self.n_docs_: int = 0

    def fit(self, tokenized_docs: List[List[str]]) -> "PaperTfidfVectorizer":
        df = Counter()
        for doc in tokenized_docs:
            df.update(set(doc))
        vocab = sorted(df.keys())
        self.vocab_ = {term: i for i, term in enumerate(vocab)}
        n = len(tokenized_docs)
        self.n_docs_ = n

        self.idf_ = np.array([math.log(n / df[t]) for t in vocab], dtype=np.float64)
        return self

    def transform(self, tokenized_docs: List[List[str]]) -> sparse.csr_matrix:

        rows, cols, data = [], [], []
        for row_idx, doc in enumerate(tokenized_docs):
            tf = Counter(doc)
            if not tf:
                continue
            max_tf = max(tf.values())
            for term, count in tf.items():
                col_idx = self.vocab_.get(term)
                if col_idx is None:
                    continue
                nf = 0.5 + 0.5 * (count / max_tf)
                w = nf * self.idf_[col_idx]
                if w != 0.0:
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(w)
        mat = sparse.csr_matrix(
            (data, (rows, cols)) if data else ([], ([], [])),
            shape=(len(tokenized_docs), len(self.vocab_)),
        )
        return _l2_normalize_rows(mat)


def _l2_normalize_rows(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    inv_norms = sparse.diags(1.0 / norms)
    return (inv_norms @ mat).tocsr()


@dataclass
class ProjectTfidfIndex:


    vectorizer: PaperTfidfVectorizer
    file_paths: List[str]
    file_matrix: sparse.csr_matrix
    function_file_offsets: Dict[str, Tuple[int, int]]
    function_matrix: sparse.csr_matrix
    report_ids: List[str]
    report_matrix: sparse.csr_matrix
    report_summary_matrix: sparse.csr_matrix
    report_row: Dict[str, int] = field(default_factory=dict)
    file_row: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.report_row = {rid: i for i, rid in enumerate(self.report_ids)}
        self.file_row = {fp: i for i, fp in enumerate(self.file_paths)}

    def phi1_all_files(self, bug_id: str) -> np.ndarray:

        r_idx = self.report_row[bug_id]
        r_vec = self.report_matrix[r_idx]

        whole_file_sims = np.asarray((r_vec @ self.file_matrix.T).todense()).ravel()

        if self.function_matrix.shape[0] > 0:
            func_sims = np.asarray((r_vec @ self.function_matrix.T).todense()).ravel()
        else:
            func_sims = np.array([])

        result = whole_file_sims.copy()
        for fp, (start, end) in self.function_file_offsets.items():
            if end > start:
                f_idx = self.file_row.get(fp)
                if f_idx is not None:
                    max_func_sim = func_sims[start:end].max()
                    if max_func_sim > result[f_idx]:
                        result[f_idx] = max_func_sim
        return result

    def phi1(self, bug_id: str, file_path: str) -> float:
        f_idx = self.file_row.get(file_path)
        if f_idx is None:
            return 0.0
        return float(self.phi1_all_files(bug_id)[f_idx])

    def report_summary_similarity(self, bug_id_a: str, bug_id_b: str) -> float:
        ia, ib = self.report_row.get(bug_id_a), self.report_row.get(bug_id_b)
        if ia is None or ib is None:
            return 0.0
        return float((self.report_summary_matrix[ia] @ self.report_summary_matrix[ib].T)[0, 0])


def build_project_tfidf_index(
    source_files: Dict[str, str],
    functions_by_file: Dict[str, List[Dict[str, str]]],
    bug_reports: Dict,
) -> ProjectTfidfIndex:

    file_paths = list(source_files.keys())
    file_tokens = [preprocess_text(source_files[fp]) for fp in file_paths]

    function_file_offsets: Dict[str, Tuple[int, int]] = {}
    function_tokens: List[List[str]] = []
    for fp in file_paths:
        start = len(function_tokens)
        for func in functions_by_file.get(fp, []):
            function_tokens.append(preprocess_text(func.get('body', '')))
        function_file_offsets[fp] = (start, len(function_tokens))

    report_ids = list(bug_reports.keys())
    report_tokens: List[List[str]] = []
    report_summary_tokens: List[List[str]] = []
    for rid in report_ids:
        report = bug_reports[rid].get('bug_report', {})
        summary = report.get('summary', '') or ''
        description = report.get('description', '') or ''
        report_tokens.append(preprocess_text(f"{summary} {description}"))
        report_summary_tokens.append(preprocess_text(summary))

    all_docs = file_tokens + function_tokens + report_tokens + report_summary_tokens
    vectorizer = PaperTfidfVectorizer().fit(all_docs)

    file_matrix = vectorizer.transform(file_tokens) if file_tokens else sparse.csr_matrix((0, len(vectorizer.vocab_)))
    function_matrix = vectorizer.transform(function_tokens) if function_tokens else sparse.csr_matrix((0, len(vectorizer.vocab_)))
    report_matrix = vectorizer.transform(report_tokens) if report_tokens else sparse.csr_matrix((0, len(vectorizer.vocab_)))
    report_summary_matrix = vectorizer.transform(report_summary_tokens) if report_summary_tokens else sparse.csr_matrix((0, len(vectorizer.vocab_)))

    return ProjectTfidfIndex(
        vectorizer=vectorizer,
        file_paths=file_paths,
        file_matrix=file_matrix,
        function_file_offsets=function_file_offsets,
        function_matrix=function_matrix,
        report_ids=report_ids,
        report_matrix=report_matrix,
        report_summary_matrix=report_summary_matrix,
    )
