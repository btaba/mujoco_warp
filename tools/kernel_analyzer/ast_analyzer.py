import ast
import logging
import inspect
from typing import Any, Dict, List, Set
import mujoco_warp as mjwarp
import warp as wp

_EXPECTED_TYPES = (
  "int",
  "float",
  "bool",
  "array",
  "array2d",
  "array2df",
  "array3d",
  "array3df",
)


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
  """Check that parameter types are present and in a permissible set."""

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


class TypeMismatchIssue:
  """Check that parameter types match the expected type."""

  def __init__(
    self,
    lineno: int,
    kernel: str,
    param_name: str,
    actual_type: str,
    expected_type: str,
    field_source: str,
  ):
    self.lineno = lineno
    self.kernel = kernel
    self.param_name = param_name
    self.actual_type = actual_type
    self.expected_type = expected_type
    self.field_source = field_source

  def __str__(self):
    return f"{self.lineno}: Kernel '{self.kernel}' parameter '{self.param_name}' has type '{self.actual_type}' but {self.field_source} field type is '{self.expected_type}'"


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


class MissingCommentIssue:
  def __init__(
    self,
    lineno: int,
    kernel: str,
    param_name: str,
    param_type: str,
    expected_comment: str,
  ):
    self.lineno = lineno
    self.kernel = kernel
    self.param_name = param_name
    self.param_type = param_type
    self.expected_comment = expected_comment

  def __str__(self):
    return f"{self.lineno}: Kernel '{self.kernel}' parameter '{self.param_name}' of type '{self.param_type}' missing '{self.expected_comment}' comment"


class WriteToReadOnlyFieldIssue:
  def __init__(self, lineno: int, kernel: str, field_name: str, field_type: str):
    self.lineno = lineno
    self.kernel = kernel
    self.field_name = field_name
    self.field_type = field_type

  def __str__(self):
    return f"{self.lineno}: Kernel '{self.kernel}' writes to {self.field_type} field '{self.field_name}' which should be read-only"

  def __eq__(self, other):
    if not isinstance(other, WriteToReadOnlyFieldIssue):
      return False
    return str(self) == str(other)

  def __hash__(self):
    return hash(str(self))


def _get_annotation_info(node):
  """Recursively analyze the type annotation and return a string representation."""
  if node is None:
    return ""

  if isinstance(node, ast.Name):
    return node.id

  if isinstance(node, ast.Attribute):
    if isinstance(node.value, ast.Name):
      return f"{node.value.id}.{node.attr}"
    elif isinstance(node.value, ast.Attribute):
      parent_info = _get_annotation_info(node.value)
      return f"{parent_info}.{node.attr}"
    return node.attr

  if isinstance(node, ast.Call):
    func_name = _get_annotation_info(node.func)
    args = []

    for arg in node.args:
      args.append(_get_annotation_info(arg))

    for kw in node.keywords:
      args.append(f"{kw.arg}={_get_annotation_info(kw.value)}")

    return f"{func_name}({', '.join(args)})"

  if isinstance(node, ast.Constant):
    return str(node.value)

  return str(type(node).__name__)


def _get_class_annotations(class_name: str, src: str) -> Dict[str, str]:
  """Return class field annotations by analyzing AST."""
  tree = ast.parse(src)

  model_annotations = {}
  for node in ast.walk(tree):
    if not isinstance(node, ast.ClassDef):
      continue
    if node.name != class_name:
      continue

    for item in node.body:
      if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
        field_name = item.target.id
        annotation_info = _get_annotation_info(item.annotation)
        model_annotations[field_name] = annotation_info

  return model_annotations


def _get_model_field_annotations() -> Dict[str, str]:
  """Return model field annotations by analyzing AST."""
  src = inspect.getsource(mjwarp.Model)
  return _get_class_annotations("Model", src)


def _get_data_field_annotations() -> Dict[str, str]:
  """Return data field annotations by analyzing AST."""
  src = inspect.getsource(mjwarp.Data)
  return _get_class_annotations("Data", src)


def _get_valid_model_fields() -> Dict[str, Any]:
  """Return valid model fields."""
  return mjwarp.Model.__annotations__


def _get_valid_data_fields() -> Dict[str, Any]:
  """Return valid data fields."""
  fields = mjwarp.Data.__annotations__
  fields_in = {k + "_in": v for k, v in fields.items()}
  fields_out = {k + "_out": v for k, v in fields.items()}
  return {**fields, **fields_in, **fields_out}


def _canonicalize_data_field_name(field_name: str) -> str:
  """Canonicalize data field name."""
  if field_name.endswith("_in"):
    return field_name[:-3]
  elif field_name.endswith("_out"):
    return field_name[:-4]
  return field_name


def _check_parameter_types(node: ast.FunctionDef, issues: List[TypeIssue]):
  """Check that parameter types are present and in a permissible set."""
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
  model_fields = _get_valid_model_fields()
  data_fields = _get_valid_data_fields()

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
  data_fields = _get_valid_data_fields()
  model_fields = _get_valid_model_fields()

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
  data_fields = _get_valid_data_fields()
  model_fields = _get_valid_model_fields()

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
  model_fields = _get_valid_model_fields()

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
  data_fields = _get_valid_data_fields()
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


def _check_argument_naming(node: ast.FunctionDef, issues: List[ArgPositionIssue]):
  _check_model_field_suffixes(node, issues)
  _check_data_field_suffixes(node, issues)


def _check_argument_positions(node: ast.FunctionDef, issues: List):
  _check_model_data_in_the_middle(node, issues)
  _check_model_fields_before_data_fields(node, issues)
  _check_data_fields_order(node, issues)


def _check_parameter_comments(
  node: ast.FunctionDef, issues: List[MissingCommentIssue], source_lines: List[str]
):
  """Check for comments on the line before the first occurrence of the first Model/Data field."""
  model_fields = _get_valid_model_fields()
  data_fields = _get_valid_data_fields()
  raw_data_fields = mjwarp.Data.__annotations__.keys()

  # Get the function body start line
  func_line = node.lineno
  func_body_start = func_line

  # Find where the actual parameters start in the source code
  for i, line in enumerate(source_lines[func_line - 1 :], func_line - 1):
    if "(" in line:
      func_body_start = i
      break

  # Track the first occurrence of each parameter category
  first_model = None
  first_regular_data = None
  first_data_in = None
  first_data_out = None

  # Find parameter positions in source code
  for param in node.args.args:
    param_name = param.arg
    param_type = None

    if param.annotation:
      if isinstance(param.annotation, ast.Call):
        param_type = param.annotation.func.attr
      else:
        param_type = param.annotation.id

    # Find the line number where this parameter appears
    param_line = None
    for i, line in enumerate(source_lines[func_body_start:], func_body_start):
      if param_name + ":" in line:
        param_line = i
        break

    # If we couldn't find the parameter, skip it
    if param_line is None:
      continue

    # Only record the first occurrence of each category
    if param_name in model_fields and first_model is None:
      first_model = (param_name, param_type, param_line, "# Model")
    elif param_name not in data_fields:
      continue  # Skip if it's not a Model or Data field
    elif param_name in raw_data_fields and first_regular_data is None:
      first_regular_data = (param_name, param_type, param_line, "# Data")
    elif param_name.endswith("_in") and first_data_in is None:
      first_data_in = (param_name, param_type, param_line, "# Data in")
    elif param_name.endswith("_out") and first_data_out is None:
      first_data_out = (param_name, param_type, param_line, "# Data out")

  # Check for comments on the lines before the first parameter of each category
  categories = [first_model, first_regular_data, first_data_in, first_data_out]

  # Sort by line number to preserve order
  categories = [c for c in categories if c is not None]
  categories.sort(key=lambda x: x[2])

  # Check each category
  for param_info in categories:
    param_name, param_type, param_line, expected_comment = param_info
    if param_line < 0 or param_line >= len(source_lines):
      continue
    # Check if the line before has the expected comment
    prev_line = source_lines[param_line - 1].strip()
    if expected_comment not in prev_line:
      issues.append(
        MissingCommentIssue(
          lineno=param_line,
          kernel=node.name,
          param_name=param_name,
          param_type=param_type,
          expected_comment=expected_comment,
        )
      )


def _check_no_writes_to_readonly_fields(
  node: ast.FunctionDef, issues: List[WriteToReadOnlyFieldIssue]
):
  """Check that the function doesn't write to Model params or Data fields with _in suffix."""
  model_fields = _get_valid_model_fields()
  data_fields = _get_valid_data_fields()

  # Track parameters that shouldn't be written to
  readonly_params = {}
  for param in node.args.args:
    param_name = param.arg
    if param_name in model_fields:
      readonly_params[param_name] = "Model"
    elif param_name.endswith("_in") and param_name[:-3] in mjwarp.Data.__annotations__:
      readonly_params[param_name] = "Data input"

  new_issues = set()
  # Visit all assignments in the function body
  for body_item in ast.walk(node):
    # Check for simple assignments
    if isinstance(body_item, ast.Assign):
      for target in body_item.targets:
        _check_target_for_readonly_writes(
          target, readonly_params, node.name, new_issues
        )
    # Check for augmented assignments (+=, -=, etc.)
    elif isinstance(body_item, ast.AugAssign):
      _check_target_for_readonly_writes(
        body_item.target, readonly_params, node.name, new_issues
      )
    # Also check for in-place operations like a[i] = value
    elif isinstance(body_item, ast.Subscript) and isinstance(body_item.ctx, ast.Store):
      _check_target_for_readonly_writes(
        body_item.value, readonly_params, node.name, new_issues
      )

  issues.extend(new_issues)


def _check_target_for_readonly_writes(target, readonly_params, kernel_name, issues):
  """Check if an assignment target is writing to a read-only parameter."""
  target_name = None

  # Simple variable name
  if isinstance(target, ast.Name):
    target_name = target.id
  # Attribute access like obj.attr
  elif isinstance(target, ast.Attribute):
    # For attribute access, we check the base object
    if isinstance(target.value, ast.Name):
      target_name = target.value.id
  # Subscript access like obj[index]
  elif isinstance(target, ast.Subscript):
    if isinstance(target.value, ast.Name):
      target_name = target.value.id

  # If we have a target name and it's in our read-only list
  if target_name and target_name in readonly_params:
    issues.add(
      WriteToReadOnlyFieldIssue(
        lineno=target.lineno if hasattr(target, "lineno") else 0,
        kernel=kernel_name,
        field_name=target_name,
        field_type=readonly_params[target_name],
      )
    )


def _check_field_type_annotations(
  node: ast.FunctionDef, issues: List[TypeMismatchIssue], source_lines: List[str]
):
  """Check that type annotations match the original Model/Data annotations exactly."""
  model_fields = _get_valid_model_fields().keys()
  data_fields = _get_valid_data_fields().keys()
  model_annotations = _get_model_field_annotations()
  data_annotations = _get_data_field_annotations()

  for param in node.args.args:
    param_name = param.arg

    if param_name in model_fields:
      expected_type = model_annotations[param_name]
      field_source = "Model"
    elif param_name in data_fields:
      param_name = _canonicalize_data_field_name(param_name)
      expected_type = data_annotations[param_name]
      field_source = "Data"
    else:
      continue

    # Skip if there's no annotation (already handled by _check_parameter_types)
    if param.annotation is None:
      continue

    # Recreate the type from the AST annotation.
    actual_type = _get_annotation_info(param.annotation)
    if actual_type != expected_type:
      issues.append(
        TypeMismatchIssue(
          lineno=node.lineno,
          kernel=node.name,
          param_name=param_name,
          actual_type=actual_type,
          expected_type=expected_type,
          field_source=field_source,
        )
      )


def analyze(code_string: str, filename: str):
  """Parses Python code and finds functions with unsorted simple parameters."""
  issues = []
  logging.info(f"Analyzing {filename}...")

  try:
    tree = ast.parse(code_string, filename=filename)
    source_lines = code_string.splitlines()

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

      _check_field_type_annotations(node, issues, source_lines)

      # Check argument naming.
      _check_argument_naming(node, issues)

      # Check argument positions.
      _check_argument_positions(node, issues)

      # Check parameter comments
      _check_parameter_comments(node, issues, source_lines)

      # Check no writes to read-only fields
      _check_no_writes_to_readonly_fields(node, issues)

  except SyntaxError as e:
    logging.error(f"Syntax error in {filename}:{e.lineno}: {e.msg}")
  except Exception as e:
    logging.error(f"Error parsing {filename}: {e}", exc_info=True)

  logging.info(f"Finished analyzing {filename}. Found {len(issues)} issues.")

  return issues
