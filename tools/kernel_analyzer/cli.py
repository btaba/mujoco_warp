import argparse
import logging
import sys
from pathlib import Path

import ast_analyzer


def main():
  parser = argparse.ArgumentParser(
    description="Check Python files for unsorted function parameters (simple check)."
  )
  parser.add_argument("files", nargs="+", help="Python files to check.")
  parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable debug logging."
  )

  args = parser.parse_args()

  log_level = logging.DEBUG if args.verbose else logging.WARNING
  logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

  for filename in args.files:
    filepath = Path(filename)

    def err(iss):
      print(f"{filepath}:" + str(iss), file=sys.stderr)

    if not filepath.is_file() or filepath.suffix != ".py":
      err("Skipping non-Python file")
      continue

    print(f"Checking: {filepath}")
    issues = []
    try:
      content = filepath.read_text(encoding="utf-8")
      issues = ast_analyzer.analyze(content, str(filepath))

      for issue in issues:
        err(issue)
    except Exception as e:
      err(f"Error processing file {filepath}: {e}")
      sys.exit(1)

  if issues:
    print("\nIssues found.")
    sys.exit(1)
  else:
    print("\nNo issues found.")
    sys.exit(0)


if __name__ == "__main__":
  main()
