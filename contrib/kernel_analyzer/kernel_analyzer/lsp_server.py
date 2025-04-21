import logging
from typing import Any, Dict, List, Optional

import ast_analyzer
from lsprotocol.types import CodeAction
from lsprotocol.types import CodeActionKind
from lsprotocol.types import CodeActionParams
from lsprotocol.types import Command
from lsprotocol.types import Diagnostic
from lsprotocol.types import DiagnosticSeverity
from lsprotocol.types import Position
from lsprotocol.types import Range
from lsprotocol.types import TextEdit
from lsprotocol.types import WorkspaceEdit
from pygls.server import LanguageServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class KernelAnalyzerLanguageServer(LanguageServer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.issues: Dict[str, List[Any]] = {}


_server = KernelAnalyzerLanguageServer("kernel-analyzer", "v0.1")


def _dict_to_lsp_range(range_dict: Dict[str, Any]) -> Optional[Range]:
  """Converts the internal 0-based dict range to LSP Range."""
  try:
    return Range(
      start=Position(
        line=range_dict["start"]["line"], character=range_dict["start"]["character"]
      ),
      end=Position(
        line=range_dict["end"]["line"], character=range_dict["end"]["character"]
      ),
    )
  except KeyError:
    logging.error(f"Error converting range dict to LSP Range: {range_dict}")
    return None


@_server.feature("textDocument/didOpen")
@_server.feature("textDocument/didChange")
@_server.feature("textDocument/didSave")
async def validate(ls: KernelAnalyzerLanguageServer, params):
  """Validate the document using core_logic.ast_analyzer."""
  text_doc = ls.workspace.get_text_document(params.text_document.uri)
  source = text_doc.source
  diagnostics: List[Diagnostic] = []  # Ensure diagnostics list is defined
  ls.issues[text_doc.uri] = []  # Clear previous issues for this file
  ast_issue_types = [x for x in dir(ast_analyzer) if x.endswith("Issue")]
  logging.info(f"Validating document: {text_doc.uri}")

  try:
    issues = ast_analyzer.analyze(source, text_doc.uri)
    logging.info(f"Analyzer found {len(issues)} issues in {text_doc.uri}")
    # store for potential future use (code actions)
    ls.issues[text_doc.uri] = issues

    for issue in issues:
      diag = Diagnostic(
        range=Range(
          start=Position(line=issue.lineno - 1, character=0),
          end=Position(line=issue.lineno, character=0),
        ),
        message=str(issue),
        severity=DiagnosticSeverity.Warning,  # Yellow underline
        code=f"KA{ast_issue_types.index(type(issue).__name__):04}",
        source="Kernel Analyzer",
      )
      diagnostics.append(diag)

  except Exception as e:
    # Log errors during validation
    logging.error(f"Error during validation for {text_doc.uri}: {e}", exc_info=True)
    # Optionally add a single general error diagnostic to the file start
    # diagnostics.append(Diagnostic(...))

  logging.info(f"Publishing {len(diagnostics)} diagnostics for {text_doc.uri}")
  ls.publish_diagnostics(text_doc.uri, diagnostics)


if __name__ == "__main__":
  logging.info("Starting Parameter Sorter Language Server...")
  _server.start_io()
