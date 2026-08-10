
import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import re

def parse_xml_streaming(xml_file, output_file):
    print(f"  Using streaming parser (this may take a few minutes for large files)...")

    bug_reports = {}
    error_count = 0

    try:
        print(f"  Reading file...")
        with open(xml_file, 'rb') as f:
            content = f.read()

        print(f"  Cleaning content ({len(content) / 1024 / 1024:.1f} MB)...")
        content_clean = bytearray()
        for i, byte in enumerate(content):
            if 9 <= byte <= 13 or 32 <= byte <= 126:
                content_clean.append(byte)
            elif byte < 128:
                content_clean.append(32)

        print(f"  Decoding and parsing...")
        content_str = bytes(content_clean).decode('utf-8', errors='ignore')

        print(f"  Extracting data (this may take a while)...")
        table_pattern = r'<table[^>]*>(.*?)</table>'
        column_pattern = r'<column\s+name="([^"]+)"[^>]*>(.*?)</column>'

        table_matches = list(re.finditer(table_pattern, content_str, re.DOTALL))
        print(f"  Found {len(table_matches)} table entries")

        for idx, table_match in enumerate(table_matches):
            if (idx + 1) % 1000 == 0:
                print(f"    Processing entry {idx + 1}/{len(table_matches)}...")

            try:
                table_content = table_match.group(1)
                current_table = {}

                for col_match in re.finditer(column_pattern, table_content, re.DOTALL):
                    name = col_match.group(1)
                    text = col_match.group(2).strip()

                    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')

                    if name == 'bug_id':
                        current_table['bug_id'] = text
                    elif name == 'summary':
                        current_table['summary'] = text
                    elif name == 'description':
                        current_table['description'] = text
                    elif name == 'commit':
                        current_table['commit'] = text
                    elif name == 'files':
                        current_table['files'] = text
                    elif name == 'report_timestamp':
                        try:
                            current_table['timestamp'] = float(text)
                        except:
                            current_table['timestamp'] = 0

                if current_table.get('bug_id') and current_table.get('commit'):
                    commit = current_table['commit']
                    bug_reports[commit] = {
                        'bug_report': {
                            'id': current_table['bug_id'],
                            'bug_id': current_table['bug_id'],
                            'summary': current_table.get('summary', ''),
                            'description': current_table.get('description', ''),
                            'timestamp': current_table.get('timestamp', 0),
                            'commit': commit,
                            'result': current_table.get('files', '')
                        }
                    }
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"    ⚠ Skipped entry {idx + 1}: {str(e)[:50]}")
                continue

        print(f"  ✓ Processed {len(table_matches)} entries, extracted {len(bug_reports)} valid bug reports")
        if error_count > 0:
            print(f"  ⚠ Skipped {error_count} problematic entries")

    except Exception as e:
        print(f"  ✗ Streaming parser error: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bug_reports, f, indent=2, ensure_ascii=False)

        print(f"✓ Parsed {len(bug_reports)} bug reports (streaming)")
        print(f"✓ Saved to {output_file}")
        return True
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
        return False


def parse_xml_to_json(xml_file, output_file):
    print(f"Parsing {xml_file}...")

    if not os.path.exists(xml_file):
        print(f"✗ Error: {xml_file} not found!")
        return False

    try:
        try:
            parser = ET.XMLParser(encoding='utf-8')
            tree = ET.parse(xml_file, parser=parser)
            root = tree.getroot()
        except (ET.ParseError, UnicodeDecodeError) as e:
            print(f"  ⚠ UTF-8 parse error: {str(e)[:100]}")
            try:
                parser = ET.XMLParser(encoding='latin-1')
                tree = ET.parse(xml_file, parser=parser)
                root = tree.getroot()
                print("  ✓ Parsed with latin-1 encoding")
            except Exception as e2:
                print(f"  ⚠ Using streaming parser to handle malformed XML...")
                try:
                    return parse_xml_streaming(xml_file, output_file)
                except Exception as e3:
                    print(f"✗ Failed to parse: {e3}")
                    print(f"  You may need to manually fix {xml_file} or skip it")
                    return False
    except Exception as e:
        print(f"✗ Error parsing XML: {e}")
        return False

    bug_reports = {}
    count = 0

    error_count = 0
    for table in root.findall('.//table'):
        try:
            bug_id = None
            summary = ''
            description = ''
            commit = ''
            files = ''
            timestamp = 0

            for column in table.findall('column'):
                try:
                    name = column.get('name')
                    text = column.text or ''

                    if text:
                        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')

                    if name == 'bug_id':
                        bug_id = text
                    elif name == 'summary':
                        summary = text
                    elif name == 'description':
                        description = text
                    elif name == 'commit':
                        commit = text
                    elif name == 'files':
                        files = text
                    elif name == 'report_timestamp':
                        try:
                            timestamp = float(text)
                        except:
                            timestamp = 0
                except Exception as e:
                    continue

            if bug_id and commit:
                bug_reports[commit] = {
                    'bug_report': {
                        'id': bug_id,
                        'bug_id': bug_id,
                        'summary': summary,
                        'description': description,
                        'timestamp': timestamp,
                        'commit': commit,
                        'result': files
                    }
                }
                count += 1
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"  ⚠ Skipped problematic entry: {e}")
            continue

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bug_reports, f, indent=2, ensure_ascii=False)

        print(f"✓ Parsed {len(bug_reports)} bug reports")
        if error_count > 0:
            print(f"  ⚠ Skipped {error_count} problematic entries")
        print(f"✓ Saved to {output_file}")
        return True
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
        return False


def main():
    dataset_folder = "dataset"

    projects = [
        ('AspectJ.xml', 'aspectj_bugs.json'),
        ('Birt.xml', 'birt_bugs.json'),
        ('Eclipse_Platform_UI.xml', 'eclipse_platform_ui_bugs.json'),
        ('JDT.xml', 'jdt_bugs.json'),
        ('SWT.xml', 'swt_bugs.json'),
        ('Tomcat.xml', 'tomcat_bugs.json'),
    ]

    print("="*60)
    print("Parsing Dataset XML Files")
    print("="*60)

    for xml_file, json_file in projects:
        xml_path = os.path.join(dataset_folder, xml_file)
        if os.path.exists(xml_path):
            parse_xml_to_json(xml_path, json_file)
        else:
            print(f"⚠ Skipping {xml_file} - not found")

    print("\n" + "="*60)
    print("✓ Parsing completed!")
    print("="*60)


if __name__ == '__main__':
    main()

