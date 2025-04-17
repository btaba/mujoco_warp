import ast
import logging
from typing import Any, Dict, List


class DefaultsParamsIssue:
  def __init__(self, lineno: int, kernel: str):
    self.lineno = lineno
    self.kernel = kernel

  def __str__(self):
    return f"{self.lineno}: Kernel '{self.kernel}' has default params"


class VarArgsIssue:
  def __init__(self, lineno: int, kernel: str):
    self.kernel = kernel
    self.lineno = lineno

  def __str__(self):
    return f"{self.lineno}: Kernel '{self.kernel}' has varargs"


class KwArgsIssue:
  def __init__(self, lineno: int, kernel: str):
    self.kernel = kernel
    self.lineno = lineno

  def __str__(self):
    return f"{self.lineno}: Kernel '{self.kernel}' has kwargs"


class TypeIssue:
  def __init__(self, lineno: int, kernel: str, param_name: str, param_type: str):
    self.lineno = lineno
    self.kernel = kernel
    self.param_name = param_name
    self.param_type = param_type

  def __str__(self):
    str = f"{self.lineno}: Kernel '{self.kernel}' param: {self.param_name} "
    if self.param_type:
      str += f"has unexpected annotation: {self.param_type}"
    else:
      str += f"missing type annotation"
    return str


def analyze(code_string: str, filename: str = "<string>"):
  """Parses Python code and finds functions with unsorted simple parameters."""
  issues = []
  logging.info(f"Analyzing {filename}...")
  try:
    tree = ast.parse(code_string, filename=filename)

    for node in ast.walk(tree):
      if isinstance(node, ast.FunctionDef):
        # Only reviewing kernel functions
        if not any(d.id == "kernel" for d in node.decorator_list):
          continue

        # defaults or kw defaults not allowed
        if node.args.defaults or node.args.kw_defaults:
          issues.append(DefaultsParamsIssue(kernel=node.name, lineno=node.lineno))

        # varargs not allowed
        if node.args.vararg:
          issues.append(VarArgsIssue(kernel=node.name, lineno=node.lineno))

        # kwargs not allowed
        if node.args.kwarg:
          issues.append(KwArgsIssue(kernel=node.name, lineno=node.lineno))

        # params must all be warp arrays
        for param in node.args.args:
          if param.annotation is None:
            issues.append(TypeIssue(node.name, node.lineno, param.arg, ""))
          elif isinstance(param.annotation, ast.Call):
            param_type = param.annotation.func.attr  # array(dtype=...)
          else:
            param_type = param.annotation.id  # array2d, array3d, etc
          if param_type not in ("array", "array2d", "array2df", "array3d", "array3df"):
            issues.append(TypeIssue(node.lineno, node.name, param.arg, param_type))
        
        # TODO(btaba): check that types match class field types

        # TODO(btaba): check that order is Model, Data in, Data out

        # TODO(btaba): check that Model fields start with #  Model coomment

        # TODO(btaba): check that fields that aren't on model or data are either at the beginning or the end of the list

        # TODO(btaba): check that 1) Model fields don't end in _in or _out, that 2) Data in fields end in in_, 3) Data out fields end in _out

        # TODO(btaba): check that you never write to Model param or an _in field

  except SyntaxError as e:
    logging.error(f"Syntax error in {filename}:{e.lineno}: {e.msg}")
  except Exception as e:
    logging.error(f"Error parsing {filename}: {e}", exc_info=True)

  logging.info(f"Finished analyzing {filename}. Found {len(issues)} issues.")

  return issues
