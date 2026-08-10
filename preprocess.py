

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import nltk
    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.stem import PorterStemmer
    try:
        _STOPWORDS = set(_nltk_stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        _STOPWORDS = set(_nltk_stopwords.words('english'))
    _STEMMER = PorterStemmer()
except ImportError as e:
    raise ImportError(
        "preprocess.py requires nltk (for stop-word removal and Porter stemming). "
        "Install with: pip install nltk"
    ) from e

try:
    from tree_sitter import Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False



_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_CAMEL_RE = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+')


def split_compound_identifier(token: str) -> List[str]:
    sub_tokens: List[str] = []
    for part in token.split('_'):
        if not part:
            continue
        sub_tokens.extend(_CAMEL_RE.findall(part))
    return sub_tokens


def preprocess_text(text: str) -> List[str]:

    if not text:
        return []

    raw_tokens = _TOKEN_RE.findall(text)
    tokens = [t for t in raw_tokens if t.lower() not in _STOPWORDS]

    expanded: List[str] = []
    for t in tokens:
        expanded.append(t)
        sub = split_compound_identifier(t)
        if len(sub) > 1:
            expanded.extend(s for s in sub if s.lower() not in _STOPWORDS)

    return [_STEMMER.stem(t.lower()) for t in expanded]


def preprocess_bug_report(summary: str, description: str) -> Tuple[List[str], List[str], List[str]]:

    summary_tokens = preprocess_text(summary or "")
    description_tokens = preprocess_text(description or "")
    combined_tokens = preprocess_text(f"{summary or ''} {description or ''}")
    return summary_tokens, description_tokens, combined_tokens



_JAVA_LANGUAGE = None
_JAVA_PARSER = None


def _get_java_parser():

    global _JAVA_LANGUAGE, _JAVA_PARSER
    if not TREE_SITTER_AVAILABLE:
        return None
    if _JAVA_PARSER is not None:
        return _JAVA_PARSER
    try:
        import tree_sitter_java as ts_java
        _JAVA_LANGUAGE = ts_java.language()
        _JAVA_PARSER = Parser(_JAVA_LANGUAGE)
        return _JAVA_PARSER
    except Exception:
        return None


def _regex_extract_functions(code: str) -> List[Dict[str, str]]:

    class_match = re.search(r'(?:class|interface|enum)\s+(\w+)', code)
    class_name = class_match.group(1) if class_match else None

    functions: List[Dict[str, str]] = []
    pattern = re.compile(
        r'(?:public|private|protected)?\s*(?:static)?\s*(?:abstract)?\s*(?:final)?\s*'
        r'(?:synchronized)?\s*(?:<[^>]*>\s*)?(?:[\w<>\[\],\s]+?)\s+(\w+)\s*\([^)]*\)\s*'
        r'(?:throws\s+[\w.]+(?:\s*,\s*[\w.]+)*)?\s*'
        r'(\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\})',
        re.MULTILINE | re.DOTALL,
    )

    for match in pattern.finditer(code):
        func_name = match.group(1)
        func_body = match.group(2)

        if class_name is not None and func_name == class_name:
            continue
        if func_name in ('static', 'if', 'for', 'while', 'switch', 'catch', 'synchronized'):
            continue

        func_signature = match.group(0).replace(func_body, '').strip()
        functions.append({'name': func_name, 'body': func_body, 'signature': func_signature})

    return functions


def extract_functions(code: str) -> List[Dict[str, str]]:

    parser = _get_java_parser()
    if parser is None:
        return _regex_extract_functions(code)

    try:
        code_bytes = bytes(code, 'utf8')
        tree = parser.parse(code_bytes)
    except Exception:
        return _regex_extract_functions(code)

    functions: List[Dict[str, str]] = []

    def get_name(node) -> str:
        for child in node.children:
            if child.type == 'identifier':
                return code_bytes[child.start_byte:child.end_byte].decode('utf8', errors='ignore')
        return 'unknown'

    def get_body(node) -> Optional[str]:
        for child in node.children:
            if child.type == 'block':
                return code_bytes[child.start_byte:child.end_byte].decode('utf8', errors='ignore')
        return None

    def traverse(node):

        if node.type == 'method_declaration':
            body = get_body(node)
            if body:
                name = get_name(node)
                body_node = next((c for c in node.children if c.type == 'block'), None)
                signature = code_bytes[node.start_byte:body_node.start_byte].decode('utf8', errors='ignore').strip() \
                    if body_node else ''
                functions.append({'name': name, 'body': body, 'signature': signature})
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return functions


def extract_functions_for_files(source_files: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:

    return {path: extract_functions(content) for path, content in source_files.items()}




def normalize_file_path(file_path: str) -> Tuple[str, str]:

    file_path = file_path.strip().replace('\\', '/')
    if file_path.startswith('/'):
        file_path = file_path[1:]
    filename = file_path.split('/')[-1]
    return file_path, filename


class GroundTruthMatcher:


    def __init__(self, source_file_paths: List[str]):
        self.by_normalized: Dict[str, str] = {}
        self.by_filename: Dict[str, List[str]] = defaultdict(list)
        self.by_classname: Dict[str, List[str]] = defaultdict(list)

        for src_path in source_file_paths:
            src_norm, src_filename = normalize_file_path(src_path)
            self.by_normalized[src_norm] = src_path
            self.by_filename[src_filename].append(src_path)
            if src_filename.endswith('.java'):
                self.by_classname[src_filename[:-5].lower()].append(src_path)

        self._by_normalized_lower = {k.lower(): v for k, v in self.by_normalized.items()}

    def _candidate_variants(self, ground_truth_path: str) -> List[str]:
        gt_cleaned = re.sub(r'/?Eclipse\s+UI/?', '/eclipseui/', ground_truth_path, flags=re.IGNORECASE)
        gt_no_eclipse = re.sub(r'/?Eclipse\s+UI/?', '/', ground_truth_path, flags=re.IGNORECASE)

        gt_path, _ = normalize_file_path(gt_cleaned)
        gt_path_no_eclipse, _ = normalize_file_path(gt_no_eclipse)

        variants = [gt_path, gt_path_no_eclipse]

        for base in (gt_path, gt_path_no_eclipse):
            idx = base.find('org/eclipse/')
            if idx >= 0:
                variants.append(base[idx:])

        for variant in list(variants):
            if 'org/eclipse/' in variant and not variant.startswith('src/'):
                variants.append(variant.replace('org/eclipse/', 'src/org/eclipse/', 1))
            if variant.startswith('src/'):
                variants.append(variant[4:])
            if 'bundles/' in variant:
                variants.append(variant.replace('bundles/', '', 1))

        seen = set()
        unique = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    def match(self, ground_truth_path: str) -> Optional[str]:
        if not ground_truth_path or not ground_truth_path.strip():
            return None

        variants = self._candidate_variants(ground_truth_path)


        for variant in variants:
            v_norm, _ = normalize_file_path(variant)
            if v_norm in self.by_normalized:
                return self.by_normalized[v_norm]
            if v_norm.lower() in self._by_normalized_lower:
                return self._by_normalized_lower[v_norm.lower()]

        for variant in variants:
            v_norm, _ = normalize_file_path(variant)
            v_lower = v_norm.lower()
            for src_norm_lower, src_path in self._by_normalized_lower.items():
                if src_norm_lower.endswith('/' + v_lower) or src_norm_lower.endswith(v_lower):
                    return src_path

        for variant in variants:
            components = [c for c in variant.split('/') if c]
            if len(components) < 2:
                continue
            for n in range(min(4, len(components)), 1, -1):
                suffix = '/'.join(components[-n:]).lower()
                for src_norm, src_path in self.by_normalized.items():
                    src_components = [c for c in src_norm.split('/') if c]
                    if len(src_components) >= n and '/'.join(src_components[-n:]).lower() == suffix:
                        return src_path


        _, gt_filename = normalize_file_path(ground_truth_path)
        if gt_filename in self.by_filename:
            return self.by_filename[gt_filename][0]
        gt_filename_lower = gt_filename.lower()
        for filename, paths in self.by_filename.items():
            if filename.lower() == gt_filename_lower:
                return paths[0]


        if gt_filename.endswith('.java'):
            candidates = self.by_classname.get(gt_filename[:-5].lower())
            if candidates:
                return candidates[0]

        return None

    def resolve_bug_reports(self, bug_reports: Dict) -> Dict[str, List[str]]:

        resolved: Dict[str, List[str]] = {}
        for bug_id, bug_data in bug_reports.items():
            report = bug_data.get('bug_report', {})
            raw_result = report.get('result', [])
            if isinstance(raw_result, str):
                raw_files = [line.strip() for line in raw_result.split('\n') if line.strip()]
            elif isinstance(raw_result, list):
                raw_files = raw_result
            else:
                raw_files = []

            matched = []
            for raw_path in raw_files:
                m = self.match(raw_path)
                if m is not None:
                    matched.append(m)
            resolved[bug_id] = matched
        return resolved
