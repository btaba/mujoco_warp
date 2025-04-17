import ast
import logging
from typing import Any, Dict, List, Set
import mujoco_warp as mjwarp

_EXPECTED_TYPES = ("array", "array2d", "array2df", "array3d", "array3df")


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
  def __init__(
    self,
    lineno: int,
    kernel: str,
    param_name: str,
    param_type: str,
    expected_types: str,
  ):
    self.lineno = lineno
    self.kernel = kernel
    self.param_name = param_name
    self.param_type = param_type
    self.expected_types = expected_types

  def __str__(self):
    str = f"{self.lineno}: Kernel '{self.kernel}' param: {self.param_name} "
    if self.param_type:
      str += f"has type '{self.param_type}', expected one of: {self.expected_types}"
    else:
      str += f"missing type annotation, expected one of: {self.expected_types}"
    return str


class ArgPositionIssue:
  def __init__(self, lineno: int, kernel: str, issue_type: str, details: str = ""):
    self.lineno = lineno
    self.kernel = kernel
    self.issue_type = issue_type
    self.details = details

  def __str__(self):
    base_msg = f"{self.lineno}: Kernel '{self.kernel}' has argument position issue: {self.issue_type}"
    if self.details:
      base_msg += f". {self.details}"
    return base_msg


class ModelFieldSuffixIssue:
  def __init__(self, lineno: int, kernel: str, param_name: str, suffix: str):
    self.lineno = lineno
    self.kernel = kernel
    self.param_name = param_name
    self.suffix = suffix

  def __str__(self):
    return f"{self.lineno}: Kernel '{self.kernel}' has Model parameter '{self.param_name}' with '{self.suffix}' suffix. Model parameters should not have _in or _out suffixes."


class DataFieldSuffixIssue:
  def __init__(
    self,
    lineno: int,
    kernel: str,
    param_name: str,
    message: str,
  ):
    self.lineno = lineno
    self.kernel = kernel
    self.param_name = param_name
    self.message = message

  def __str__(self):
    return f"{self.lineno}: Kernel '{self.kernel}' has Data parameter '{self.param_name}' issue: {self.message}"


def get_valid_model_fields() -> Dict[str, Any]:
  """Return valid model fields."""
  return mjwarp.Model.__annotations__


def get_valid_data_fields() -> Dict[str, Any]:
  """Return valid data fields."""
  fields = mjwarp.Data.__annotations__
  fields_in = {k + "_in": v for k, v in fields.items()}
  fields_out = {k + "_out": v for k, v in fields.items()}
  return {**fields, **fields_in, **fields_out}


def _check_parameter_types(node: ast.FunctionDef, issues: List[TypeIssue]):
  for param in node.args.args:
    if param.annotation is None:
      issues.append(
        TypeIssue(node.lineno, node.name, param.arg, "", str(_EXPECTED_TYPES))
      )
    elif isinstance(param.annotation, ast.Call):
      param_type = param.annotation.func.attr  # array(dtype=...)
    else:
      param_type = param.annotation.id  # array2d, array3d, etc
    if param_type not in _EXPECTED_TYPES:
      issues.append(
        TypeIssue(node.lineno, node.name, param.arg, param_type, str(_EXPECTED_TYPES))
      )


def _check_model_data_in_the_middle(
  node: ast.FunctionDef, issues: List[ArgPositionIssue]
):
  model_fields = get_valid_model_fields()
  data_fields = get_valid_data_fields()

  # Check that Model/Data fields are in the middle, other parameters at beginning or end.
  model_indices, data_indices, other_indices = [], [], []
  total_params = len(node.args.args)

  for i, param in enumerate(node.args.args):
    param_name = param.arg
    # Check if this parameter name is a field in Model or Data
    if param_name in model_fields:
      model_indices.append(i)
    elif param_name in data_fields:
      data_indices.append(i)
    else:
      other_indices.append(i)

  model_data_indices = model_indices + data_indices
  if model_data_indices and other_indices:
    # Get the range of Model/Data indices.
    min_model_data = min(model_data_indices)
    max_model_data = max(model_data_indices)

    # Check each other parameter to see if it's in the middle.
    for i in other_indices:
      if min_model_data < i < max_model_data:
        issues.append(
          ArgPositionIssue(
            lineno=node.lineno,
            kernel=node.name,
            issue_type="Non-Model/Data parameter in the middle",
            details=f"Parameter '{node.args.args[i].arg}' is between Model/Data parameters",
          )
        )


def _check_model_fields_before_data_fields(
  node: ast.FunctionDef, issues: List[ArgPositionIssue]
):
  # Check that Model fields come before Data fields.
  data_fields = get_valid_data_fields()
  model_fields = get_valid_model_fields()

  model_indices, data_indices = [], []
  for i, param in enumerate(node.args.args):
    param_name = param.arg
    if param_name in model_fields:
      model_indices.append(i)
    elif param_name in data_fields:
      data_indices.append(i)

  if model_indices and data_indices:
    if max(model_indices) > min(data_indices):
      issues.append(
        ArgPositionIssue(
          lineno=node.lineno,
          kernel=node.name,
          issue_type="Model fields after Data fields",
          details="Model parameters should come before Data parameters",
        )
      )


def _check_data_fields_order(node: ast.FunctionDef, issues: List[ArgPositionIssue]):
  # Check that regular Data fields come before Data _in fields, which come before Data _out fields.
  data_fields = get_valid_data_fields()
  model_fields = get_valid_model_fields()

  data_regular_indices, data_in_indices, data_out_indices = [], [], []
  for i, param in enumerate(node.args.args):
    param_name = param.arg
    if param_name in data_fields:
      if param_name.endswith("_in"):
        data_in_indices.append(i)
      elif param_name.endswith("_out"):
        data_out_indices.append(i)
      else:
        data_regular_indices.append(i)

  # Check order: regular data params -> data_in params -> data_out params.
  # If we have regular and _in parameters, check regular comes before _in.
  if data_regular_indices and data_in_indices:
    max_regular_idx = max(data_regular_indices)
    min_in_idx = min(data_in_indices)
    if max_regular_idx > min_in_idx:
      issues.append(
        ArgPositionIssue(
          lineno=node.lineno,
          kernel=node.name,
          issue_type="Regular Data fields after Data _in fields",
          details="Regular Data parameters should come before parameters with _in suffix",
        )
      )

  # If we have _in and _out parameters, check _in comes before _out.
  if data_in_indices and data_out_indices:
    max_in_idx = max(data_in_indices)
    min_out_idx = min(data_out_indices)
    if max_in_idx > min_out_idx:
      issues.append(
        ArgPositionIssue(
          lineno=node.lineno,
          kernel=node.name,
          issue_type="Data _in fields after Data _out fields",
          details="Parameters with _in suffix should come before parameters with _out suffix",
        )
      )

  # If we have regular and _out parameters, check regular comes before _out.
  if data_regular_indices and data_out_indices:
    max_regular_idx = max(data_regular_indices)
    min_out_idx = min(data_out_indices)
    if max_regular_idx > min_out_idx:
      issues.append(
        ArgPositionIssue(
          lineno=node.lineno,
          kernel=node.name,
          issue_type="Regular Data fields after Data _out fields",
          details="Regular Data parameters should come before parameters with _out suffix",
        )
      )


def _check_model_field_suffixes(
  node: ast.FunctionDef, issues: List[ModelFieldSuffixIssue]
):
  """Check that Model fields don't end with _in or _out suffixes."""
  model_fields = get_valid_model_fields()

  for param in node.args.args:
    param_name = param.arg
    if param_name.endswith("_in") and param_name[:-3] in model_fields:
      issues.append(
        ModelFieldSuffixIssue(
          lineno=node.lineno, kernel=node.name, param_name=param_name, suffix="_in"
        )
      )
    if param_name.endswith("_out") and param_name[:-3] in model_fields:
      issues.append(
        ModelFieldSuffixIssue(
          lineno=node.lineno, kernel=node.name, param_name=param_name, suffix="_out"
        )
      )


def _check_data_field_suffixes(
  node: ast.FunctionDef, issues: List[DataFieldSuffixIssue]
):
  """Check that Data fields either use the original name or end with _in or _out."""
  data_fields = get_valid_data_fields()
  raw_data_fields = mjwarp.Data.__annotations__.keys()

  for param in node.args.args:
    param_name = param.arg

    # Skip if the parameter is a valid data field (original name or with suffix)
    if param_name in data_fields:
      continue

    # Check if it might be a data field with an invalid suffix
    for field in raw_data_fields:
      # If the parameter name starts with a valid data field name but isn't in valid_data_fields
      if param_name.startswith(field + "_") and not (
        param_name.endswith("_in") or param_name.endswith("_out")
      ):
        issues.append(
          DataFieldSuffixIssue(
            lineno=node.lineno,
            kernel=node.name,
            param_name=param_name,
            message=f"Invalid suffix. Data fields should use original name or end with _in or _out",
          )
        )


def _check_argument_positions(node: ast.FunctionDef, issues: List):
  _check_model_data_in_the_middle(node, issues)
  _check_model_fields_before_data_fields(node, issues)
  _check_data_fields_order(node, issues)
  _check_model_field_suffixes(node, issues)
  _check_data_field_suffixes(node, issues)


def analyze(code_string: str, filename: str = "<string>"):
  """Parses Python code and finds functions with unsorted simple parameters."""
  issues = []
  logging.info(f"Analyzing {filename}...")

  try:
    tree = ast.parse(code_string, filename=filename)

    for node in ast.walk(tree):
      # Only review kernel functions.
      if not isinstance(node, ast.FunctionDef):
        continue
      if not any(d.id == "kernel" for d in node.decorator_list):
        continue

      # Defaults or kw defaults not allowed.
      if node.args.defaults or node.args.kw_defaults:
        issues.append(DefaultsParamsIssue(kernel=node.name, lineno=node.lineno))

      # varargs not allowed.
      if node.args.vararg:
        issues.append(VarArgsIssue(kernel=node.name, lineno=node.lineno))

      # kwargs not allowed.
      if node.args.kwarg:
        issues.append(KwArgsIssue(kernel=node.name, lineno=node.lineno))

      # Check parameter type annotations.
      _check_parameter_types(node, issues)

      # Check argument positions.
      _check_argument_positions(node, issues)

      # TODO(btaba): check that Model fields start with #  Model coomment
      # TODO(btaba): check that types match class field types
      # TODO(btaba): check that you never write to Model param or an _in field

  except SyntaxError as e:
    logging.error(f"Syntax error in {filename}:{e.lineno}: {e.msg}")
  except Exception as e:
    logging.error(f"Error parsing {filename}: {e}", exc_info=True)

  logging.info(f"Finished analyzing {filename}. Found {len(issues)} issues.")

  return issues
