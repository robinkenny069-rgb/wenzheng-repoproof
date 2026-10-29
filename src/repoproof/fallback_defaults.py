"""Bounded AST recognition of a local class's option fallback declaration.

Recognizes ``self.options.get('key') or self.default`` and
``self.options.get('key', self.default)``. It never imports a source module,
executes a getter, resolves decorators or infers a number's unit. A resolved
result describes these expressions under ordinary dictionary-get semantics;
it does not establish a property's runtime behavior or an installed API default.
"""
from __future__ import annotations

import ast
import io
import keyword
import tokenize

from .contract_graph import Source, TraceUnavailable, dotted, literal


SCOPE = "local_class_option_fallback_expression_only"
ASSUMPTIONS = [
    "Source text is inspected without importing modules or executing getters.",
    "get is interpreted as ordinary dict.get; the receiver's runtime type and method are not proved.",
    "Only local declarations and visible direct writes are checked; inherited behavior, aliases, external mutation and installed environment are not proved.",
    "Decorator behavior, descriptor access, instance overrides and runtime API defaults are not proved.",
    "Units are not inferred from numeric literals or comments.",
]


def _parse_source(source):
    """Allow historical async import paths / call keywords, width intact.

    Only dotted import-path names and call keyword names are changed. Strings,
    comments, variables, function parameters and async syntax are untouched. AST
    source spans still index the original text, which is used for every trace.
    """
    try:
        return ast.parse(source.text, filename=source.path), False
    except SyntaxError:
        tokens = list(tokenize.generate_tokens(io.StringIO(source.text).readline))
        replacements = []
        mode, previous, before_previous = None, None, None
        brackets = []
        meaningful = [token for token in tokens if token.type not in
                      (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)]
        for index, token in enumerate(meaningful):
            if token.type in (tokenize.NEWLINE, tokenize.ENDMARKER) or token.string == ';':
                mode, previous, before_previous = None, None, None
                continue
            if token.type == tokenize.NAME:
                if token.string == 'from':
                    mode = 'from_path'
                elif token.string == 'import':
                    mode = 'from_names' if mode == 'from_path' else 'import_path'
                elif token.string == 'async' and mode in ('from_path', 'import_path') and previous and previous.string == '.':
                    replacements.append(token)
                elif (token.string == 'async' and brackets and brackets[-1] == ('(', True)
                      and previous and previous.string in ('(', ',')
                      and index + 1 < len(meaningful) and meaningful[index + 1].string == '='):
                    replacements.append(token)
            if token.string in ('(', '[', '{'):
                is_call = (token.string == '(' and previous and previous.type == tokenize.NAME
                           and not keyword.iskeyword(previous.string)
                           and (not before_previous or before_previous.string not in ('def', 'class')))
                brackets.append((token.string, bool(is_call)))
            elif token.string in (')', ']', '}') and brackets:
                brackets.pop()
            before_previous, previous = previous, token
        if not replacements:
            raise
        lines = source.text.split('\n')
        for token in replacements:
            row, column = token.start
            lines[row - 1] = lines[row - 1][:column] + 'asyn_' + lines[row - 1][column + 5:]
        return ast.parse('\n'.join(lines), filename=source.path), True


class _Bindings(ast.NodeVisitor):
    """Bindings in one namespace, including conditional writes, excluding scopes."""

    def __init__(self, body):
        self.names: dict[str, list[ast.AST]] = {}
        for node in body:
            self.visit(node)

    def add(self, name, node):
        self.names.setdefault(name, []).append(node)

    def targets(self, target, node):
        for child in ast.walk(target):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                self.add(child.id, node)

    def visit_Assign(self, node):
        for target in node.targets:
            self.targets(target, node)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self.targets(node.target, node)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        self.targets(node.target, node)
        self.visit(node.value)

    def visit_NamedExpr(self, node):
        self.targets(node.target, node)
        self.visit(node.value)

    def visit_Delete(self, node):
        for target in node.targets:
            self.targets(target, node)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.add(node.id, node)

    def visit_Import(self, node):
        for alias in node.names:
            self.add(alias.asname or alias.name.split('.')[0], node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.add(alias.asname or alias.name, node)

    def visit_FunctionDef(self, node):
        self.add(node.name, node)
        # Defaults and decorators are evaluated in the surrounding namespace.
        for expression in [*node.decorator_list, *node.args.defaults,
                           *[v for v in node.args.kw_defaults if v is not None]]:
            self.visit(expression)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.add(node.name, node)
        for expression in [*node.decorator_list, *node.bases, *[k.value for k in node.keywords]]:
            self.visit(expression)

    def visit_Lambda(self, node):
        pass

    def visit_ExceptHandler(self, node):
        if node.name:
            self.add(node.name, node)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.add(node.name, node)
        self.generic_visit(node)

    def visit_MatchStar(self, node):
        if node.name:
            self.add(node.name, node)

    def visit_MatchMapping(self, node):
        if node.rest:
            self.add(node.rest, node)
        self.generic_visit(node)

    def visit_ListComp(self, node):
        # Comprehension targets have their own scope; walrus bindings do not.
        for child in ast.walk(node):
            if isinstance(child, ast.NamedExpr):
                self.targets(child.target, child)

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


def _mark(source, node, kind):
    return {
        "kind": kind, "module": source.module, "path": source.path,
        "line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno),
        "text": ast.get_source_segment(source.text, node) or "",
        # AST line numbers count LF, not Unicode separators in string literals.
        "source_line": source.text.split('\n')[node.lineno - 1].rstrip('\r'),
    }


def _direct_attr(node, receiver):
    return (node.attr if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == receiver else None)


def _written_member(node, receivers):
    """First attribute below a recognized receiver, including subscript writes."""
    member = None
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            member = node.attr
        node = node.value
    return member if isinstance(node, ast.Name) and node.id in receivers else None


class FallbackDefaults:
    """Resolve a qualified class name (or a unique simple name) in given sources."""

    def __init__(self, sources: list[Source]):
        self.sources = {source.module: source for source in sources}
        if len(self.sources) != len(sources):
            raise ValueError("Duplicate module mapping")
        self.trees = {}
        self.legacy_imports = set()
        self.parse_errors = set()
        for source in sources:
            try:
                tree, legacy = _parse_source(source)
                self.trees[source.module] = tree
                if legacy:
                    self.legacy_imports.add(source.module)
            except (SyntaxError, tokenize.TokenError, IndentationError):
                self.parse_errors.add(source.module)

    def method_names(self, class_name: str) -> list[str]:
        """List directly declared methods; no successful-resolution filtering."""
        if self.parse_errors:
            return []
        matches = [(module, tree, node) for module, tree in self.trees.items()
                   for node in tree.body if isinstance(node, ast.ClassDef)
                   and class_name in (node.name, f"{module}.{node.name}")]
        if len(matches) != 1:
            return []
        _, tree, cls = matches[0]
        if len(_Bindings(tree.body).names.get(cls.name, [])) != 1:
            return []
        return list(dict.fromkeys(node.name for node in cls.body
                                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))))

    def resolve(self, class_name: str, property_name: str) -> dict:
        result = {"status": "abstain", "class_name": class_name,
                  "property_name": property_name, "scope": SCOPE,
                  "assumptions": list(ASSUMPTIONS), "trace": []}
        try:
            if self.parse_errors:
                raise TraceUnavailable("source_syntax_unsupported")
            candidates = []
            for module, tree in self.trees.items():
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and class_name in (node.name, f"{module}.{node.name}"):
                        candidates.append((self.sources[module], tree, node))
            if len(candidates) != 1:
                raise TraceUnavailable("class_missing_conditional_or_ambiguous")
            source, tree, cls = candidates[0]
            qualified = f"{source.module}.{cls.name}"
            result["class_name"] = qualified
            if source.module in self.legacy_imports:
                result["assumptions"].append("Historical async tokens in dotted import paths or call keyword names were replaced with an equal-width parsing placeholder; original source text and spans are preserved.")
            module_bindings = _Bindings(tree.body).names
            if len(module_bindings.get(cls.name, [])) != 1 or '*' in module_bindings:
                raise TraceUnavailable("class_binding_reassigned_or_opaque")
            if cls.decorator_list or cls.keywords:
                raise TraceUnavailable("class_decorator_or_metaclass")
            members = _Bindings(cls.body).names
            if any(name in members for name in ("__getattribute__", "__getattr__", "__setattr__", "__delattr__")):
                raise TraceUnavailable("custom_attribute_dispatch")
            getters = members.get(property_name, [])
            if len(getters) != 1 or getters[0] not in cls.body or not isinstance(getters[0], ast.FunctionDef):
                raise TraceUnavailable("getter_missing_conditional_or_rebound")
            getter = getters[0]
            args = getter.args
            positional = [*args.posonlyargs, *args.args]
            if len(positional) != 1 or args.defaults or args.kwonlyargs or args.vararg or args.kwarg:
                raise TraceUnavailable("unsupported_getter_signature")
            receiver = positional[0].arg
            body = list(getter.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body.pop(0)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                raise TraceUnavailable("getter_requires_single_return")
            statement = body[0]
            expression = statement.value
            if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.Or) and len(expression.values) == 2:
                call, fallback = expression.values
                kind, argc = "truthy_or", 1
            elif isinstance(expression, ast.Call) and len(expression.args) == 2:
                call, fallback = expression, expression.args[1]
                kind, argc = "missing_key", 2
            else:
                raise TraceUnavailable("unsupported_fallback_expression")
            if (not isinstance(call, ast.Call) or call.keywords or len(call.args) != argc
                    or not isinstance(call.func, ast.Attribute) or call.func.attr != "get"):
                raise TraceUnavailable("unsupported_get_call")
            options_member = _direct_attr(call.func.value, receiver)
            fallback_member = _direct_attr(fallback, receiver)
            if not options_member or not fallback_member:
                raise TraceUnavailable("get_and_fallback_require_direct_receiver_attributes")
            if not isinstance(call.args[0], ast.Constant) or type(call.args[0].value) is not str:
                raise TraceUnavailable("option_key_requires_string_literal")
            bindings = members.get(fallback_member, [])
            if len(bindings) != 1 or bindings[0] not in cls.body:
                raise TraceUnavailable("fallback_binding_missing_conditional_or_rebound")
            binding = bindings[0]
            if isinstance(binding, ast.Assign) and len(binding.targets) == 1 and isinstance(binding.targets[0], ast.Name):
                value_node = binding.value
            elif isinstance(binding, ast.AnnAssign) and isinstance(binding.target, ast.Name) and binding.value is not None:
                value_node = binding.value
            else:
                raise TraceUnavailable("fallback_requires_single_literal_assignment")
            value = literal(value_node)
            watched = {fallback_member, property_name, options_member}
            self._reject_writes(source, cls, watched)
            if cls.bases:
                result["assumptions"].append("Base classes are present but not traversed; only this class's local fallback declaration is resolved.")
            decorators = [ast.get_source_segment(source.text, d) or "" for d in getter.decorator_list]
            if decorators:
                result["assumptions"].append("Getter decorators are recorded verbatim and are not interpreted or executed.")
            trace = [_mark(source, d, "uninterpreted_decorator") for d in getter.decorator_list]
            trace.extend([_mark(source, statement, "option_fallback_expression"),
                          _mark(source, binding, "local_class_fallback_binding")])
            semantics = (
                "Ordinary dict.get with no default returns None for an omitted key; or then selects the fallback. Present falsy values, including 0, also select it."
                if kind == "truthy_or" else
                "Ordinary dict.get selects the fallback only for an omitted key. Present values, including 0, None and False, are preserved."
            )
            result.update(status="resolved", value=value, value_type=type(value).__name__,
                          option_key=call.args[0].value, options_binding=f"{receiver}.{options_member}",
                          fallback_binding=f"{qualified}.{fallback_member}", fallback_kind=kind,
                          fallback_semantics=semantics, zero_uses_fallback=kind == "truthy_or",
                          decorated=bool(decorators), decorators=decorators,
                          bases=[dotted(base) or ast.get_source_segment(source.text, base) for base in cls.bases],
                          unit=None, trace=trace)
        except TraceUnavailable as exc:
            result["reason"] = str(exc)
        return result

    def _reject_writes(self, source, cls, watched):
        # Local methods may receive a name other than self/cls.
        receivers = {cls.name}
        for member in cls.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [*member.args.posonlyargs, *member.args.args]
                if args:
                    receivers.add(args[0].arg)
        for module, tree in self.trees.items():
            for node in ast.walk(tree):
                # Instance names only have meaning inside the selected class.
                in_class = module == source.module and cls.lineno <= getattr(node, 'lineno', 0) <= cls.end_lineno
                owners = receivers if in_class else {cls.name}
                if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    member = _written_member(node, owners)
                    if member in watched or member == '__dict__':
                        raise TraceUnavailable("visible_attribute_write")
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ('setattr', 'delattr') and len(node.args) >= 2:
                        if isinstance(node.args[0], ast.Name) and node.args[0].id in owners:
                            key = node.args[1]
                            if not isinstance(key, ast.Constant) or key.value in watched or key.value == '__dict__':
                                raise TraceUnavailable("visible_dynamic_attribute_write")
                    if isinstance(node.func, ast.Attribute) and _written_member(node.func.value, owners) == '__dict__':
                        raise TraceUnavailable("visible_dynamic_attribute_write")
