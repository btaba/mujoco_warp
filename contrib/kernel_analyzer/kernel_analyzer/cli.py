# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
from pathlib import Path

import ast_analyzer
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

_VERBOSE = flags.DEFINE_bool("verbose", False, "Enable debug logging.")
_FILES = flags.DEFINE_multi_string("files", [], "Python files to check.")
_OUTPUT = flags.DEFINE_enum(
  "output", "console", ["console", "github"], "Analyzer output format."
)


def main(argv):
  del argv  # Unused.

  log_level = logging.DEBUG if _VERBOSE.value else logging.WARNING
  logging.set_verbosity(log_level)

  if not _FILES.value:
    logging.error("No files specified. Use --files to specify files to check.")
    sys.exit(1)

  issues = []
  for filename in _FILES.value:
    filepath = Path(filename)

    def err_console(iss):
      print(f"{filepath}:{iss.lineno}:{iss}", file=sys.stderr)

    def err_github(iss):
      print(
        f"::error title=Kernel Analyzer,file={filepath},line={iss.lineno + 1}::{iss}"
      )

    err = {"console": err_console, "github": err_github}[_OUTPUT.value]

    if not filepath.is_file() or filepath.suffix != ".py":
      err("Skipping non-Python file")
      continue

    logging.info(f"Checking: {filepath}")
    try:
      content = filepath.read_text(encoding="utf-8")
      file_issues = ast_analyzer.analyze(content, str(filepath))
      issues.extend(file_issues)

      for issue in file_issues:
        err(issue)
    except Exception as e:
      err(f"Error processing file {filepath}: {e}")
      sys.exit(1)

  if issues:
    logging.error("\nIssues found.")
    sys.exit(1)
  else:
    logging.info("\nNo issues found.")
    sys.exit(0)


if __name__ == "__main__":
  app.run(main)
