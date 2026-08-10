
import os
import json
import pickle
import zipfile
import tarfile
import shutil
from typing import Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileExtractor:

    @staticmethod
    def extract_archive(archive_path: str, extract_to: str, remove_after: bool = False) -> str:
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        extract_to.mkdir(parents=True, exist_ok=True)

        try:
            if archive_path.suffix == '.zip':
                logger.info(f"Extracting ZIP archive: {archive_path}")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    logger.info(f"Extracted to: {extract_to}")

            elif archive_path.suffix in ['.tar', '.gz'] or '.tar.gz' in archive_path.name:
                logger.info(f"Extracting TAR archive: {archive_path}")
                mode = 'r:gz' if '.gz' in archive_path.name else 'r'
                with tarfile.open(archive_path, mode) as tar_ref:
                    tar_ref.extractall(extract_to)
                    logger.info(f"Extracted to: {extract_to}")

            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

            if remove_after:
                archive_path.unlink()
                logger.info(f"Removed archive: {archive_path}")

            return str(extract_to)

        except Exception as e:
            logger.error(f"Error extracting archive {archive_path}: {e}")
            raise

    @staticmethod
    def validate_file(file_path: str, required: bool = True) -> bool:
        file_path = Path(file_path)

        if not file_path.exists():
            if required:
                raise FileNotFoundError(f"Required file not found: {file_path}")
            return False

        if not file_path.is_file():
            if required:
                raise ValueError(f"Path is not a file: {file_path}")
            return False

        if not os.access(file_path, os.R_OK):
            if required:
                raise PermissionError(f"Cannot read file: {file_path}")
            return False

        return True

    @staticmethod
    def validate_directory(dir_path: str, required: bool = True) -> bool:
        dir_path = Path(dir_path)

        if not dir_path.exists():
            if required:
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
            return False

        if not dir_path.is_dir():
            if required:
                raise ValueError(f"Path is not a directory: {dir_path}")
            return False

        if not os.access(dir_path, os.R_OK):
            if required:
                raise PermissionError(f"Cannot read directory: {dir_path}")
            return False

        return True


def load_source_files_robust(project_path: str,
                             file_extensions: list = ['.java'],
                             max_file_size: int = 10 * 1024 * 1024) -> Dict[str, str]:
    source_files = {}
    project_path = Path(project_path)

    if not project_path.exists():
        raise FileNotFoundError(f"Project path not found: {project_path}")

    if project_path.is_file() and project_path.suffix in ['.zip', '.tar', '.gz']:
        logger.info(f"Detected archive file: {project_path}")
        extract_dir = project_path.parent / f"{project_path.stem}_extracted"

        if not extract_dir.exists():
            FileExtractor.extract_archive(str(project_path), str(extract_dir))
        else:
            logger.info(f"Using existing extracted directory: {extract_dir}")

        project_path = extract_dir

    if project_path.is_file() and project_path.suffix in ['.pkl', '.pickle']:
        logger.info(f"Loading from pickle file: {project_path}")
        try:
            with open(project_path, 'rb') as f:
                source_files = pickle.load(f)
            logger.info(f"Loaded {len(source_files)} files from pickle")
            return source_files
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            raise

    if project_path.is_file() and project_path.suffix == '.json':
        logger.info(f"Loading from JSON file: {project_path}")
        try:
            with open(project_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    source_files = data
                else:
                    raise ValueError("JSON file must contain a dictionary")
            logger.info(f"Loaded {len(source_files)} files from JSON")
            return source_files
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise

    if project_path.is_dir():
        logger.info(f"Loading files from directory: {project_path}")
        file_count = 0
        error_count = 0

        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                file_path = Path(root) / file

                if not any(file.endswith(ext) for ext in file_extensions):
                    continue

                try:
                    file_size = file_path.stat().st_size
                    if file_size > max_file_size:
                        logger.warning(f"Skipping large file: {file_path} ({file_size} bytes)")
                        continue
                except Exception as e:
                    logger.warning(f"Cannot get file size for {file_path}: {e}")
                    continue

                try:
                    rel_path = file_path.relative_to(project_path)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    source_files[str(rel_path)] = content
                    file_count += 1

                    if file_count % 1000 == 0:
                        logger.info(f"Loaded {file_count} files...")

                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
                    error_count += 1
                    continue

        logger.info(f"Successfully loaded {file_count} files (errors: {error_count})")
        return source_files

    raise ValueError(f"Unsupported project path type: {project_path}")


def load_bug_reports_robust(bug_reports_path: str) -> Dict:
    bug_reports_path = Path(bug_reports_path)

    if not bug_reports_path.exists():
        raise FileNotFoundError(f"Bug reports file not found: {bug_reports_path}")

    if bug_reports_path.suffix == '.json':
        logger.info(f"Loading bug reports from JSON: {bug_reports_path}")
        try:
            with open(bug_reports_path, 'r', encoding='utf-8') as f:
                bug_reports = json.load(f)
            logger.info(f"Loaded {len(bug_reports)} bug reports")
            return bug_reports
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise

    elif bug_reports_path.suffix in ['.pkl', '.pickle']:
        logger.info(f"Loading bug reports from pickle: {bug_reports_path}")
        try:
            with open(bug_reports_path, 'rb') as f:
                bug_reports = pickle.load(f)
            logger.info(f"Loaded {len(bug_reports)} bug reports")
            return bug_reports
        except Exception as e:
            logger.error(f"Error loading pickle: {e}")
            raise

    elif bug_reports_path.suffix == '.xml':
        logger.error(
            "XML format is not loaded directly. Run scripts/parse_dataset.py first "
            "to convert dataset/<Project>.xml into <project>_bugs.json, then pass that JSON here."
        )
        raise ValueError("XML format not directly supported. Run scripts/parse_dataset.py to convert to JSON first.")

    else:
        raise ValueError(f"Unsupported bug reports format: {bug_reports_path.suffix}")


def save_data_robust(data: Dict, output_path: str, format: str = 'json'):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format in ['pickle', 'pkl']:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Successfully saved data to: {output_path}")

    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

