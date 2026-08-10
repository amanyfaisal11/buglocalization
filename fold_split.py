

from collections import OrderedDict
from typing import Dict, List, Tuple

PROJECT_FOLD_COUNTS = {
    'aspectj': 2,
    'tomcat': 2,
    'birt': 8,
    'swt': 8,
    'eclipse_platform_ui': 12,
    'jdt': 12,
}


def infer_project_key(name: str) -> str:

    n = name.lower()
    for key in PROJECT_FOLD_COUNTS:
        if key in n or key.replace('_', '') in n.replace('_', ''):
            return key
    aliases = {
        'eclipse': 'eclipse_platform_ui',
        'eclipseui': 'eclipse_platform_ui',
        'platform_ui': 'eclipse_platform_ui',
    }
    for alias, key in aliases.items():
        if alias in n:
            return key
    raise ValueError(
        f"Could not infer project fold count for '{name}'. "
        f"Known projects: {sorted(PROJECT_FOLD_COUNTS)}"
    )


def num_folds_for_project(project_name: str) -> int:
    return PROJECT_FOLD_COUNTS[infer_project_key(project_name)]


def consecutive_folds_split(bug_reports: Dict, project_name: str) -> List[Tuple[Dict, Dict]]:

    k = num_folds_for_project(project_name)

    sorted_reports = OrderedDict(
        sorted(bug_reports.items(), key=lambda x: x[1].get('bug_report', {}).get('timestamp', 0))
    )
    items = list(sorted_reports.items())
    total = len(items)

    if total < k:
        raise ValueError(
            f"Project '{project_name}' has only {total} bug reports but requires "
            f"{k} folds; cannot honor the paper's fixed fold count."
        )

    fold_size = total // k
    folds: List[Dict] = []
    for i in range(k):
        start = i * fold_size
        end = total if i == k - 1 else (i + 1) * fold_size
        folds.append(dict(items[start:end]))

    print(f"Consecutive-fold split for '{project_name}': {total} reports -> {k} folds of ~{fold_size}")
    for i, fold in enumerate(folds, start=1):
        print(f"  Fold {i}: {len(fold)} reports")

    pairs = [(folds[i], folds[i + 1]) for i in range(k - 1)]
    print(f"  -> {len(pairs)} train/test pairs (train fold n, test fold n+1)")
    return pairs


def get_fold_info(bug_reports: Dict, project_name: str) -> Dict:
    k = num_folds_for_project(project_name)
    total = len(bug_reports)
    fold_size = total // k
    return {
        'project': project_name,
        'num_folds': k,
        'total_reports': total,
        'fold_size': fold_size,
        'num_train_test_pairs': max(0, k - 1),
    }
