"""Bounded static tracing of constructor defaults through explicit forwarding.

No source imports or execution. Supports unique top-level classes, explicit
imports/exports, single inheritance, zero-argument super().__init__, simple
class-returning instance factories, and aliases of **kwargs. Unknown mutations,
escapes, branches affecting the parameter, dynamic targets, and ambiguity abstain.
This is a declared-contract trace under an explicit source snapshot, not proof
of arbitrary runtime Python behavior or a reconstruction of an installed env.
"""
from __future__ import annotations
import ast
import math
from dataclasses import dataclass
from typing import Any


class TraceUnavailable(ValueError):
    pass


@dataclass
class Source:
    module: str
    path: str
    text: str
    is_package: bool = False


@dataclass
class ClassInfo:
    qualified: str
    source: Source
    node: ast.ClassDef


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def literal(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise TraceUnavailable("non_literal_value") from exc
    if value is not None and type(value) not in (str, bool, int, float):
        raise TraceUnavailable("unsupported_literal_type")
    if type(value) is float and not math.isfinite(value):
        raise TraceUnavailable("non_finite_literal")
    return value


def mark(info: ClassInfo, node, kind: str, **extra) -> dict:
    return {"kind": kind, "module": info.source.module, "path": info.source.path,
            "line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno),
            "text": (ast.get_source_segment(info.source.text, node) or "")[:1500],
            "source_line": info.source.text.splitlines()[node.lineno - 1], **extra}


class ContractGraph:
    def __init__(self, sources: list[Source]):
        self.sources = {source.module: source for source in sources}
        if len(self.sources) != len(sources):
            raise ValueError("Duplicate module mapping")
        self.bindings: dict[str, dict[str, tuple[str, ast.AST] | None]] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.opaque_modules = set()
        for source in sources:
            tree = ast.parse(source.text, filename=source.path)
            if any(isinstance(node, ast.ImportFrom) and any(alias.name == '*' for alias in node.names) for node in ast.walk(tree)):
                self.opaque_modules.add(source.module)
            bindings = self.bindings[source.module] = {}
            def bind(name, target, node):
                bindings[name] = None if name in bindings else (target, node)
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        bind(alias.asname or alias.name.split('.')[0], alias.name if alias.asname else alias.name.split('.')[0], node)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        package = source.module.split('.') if source.is_package else source.module.split('.')[:-1]
                        prefix = package[:len(package) - node.level + 1]
                        if node.level > len(package):
                            raise TraceUnavailable("relative_import_outside_snapshot")
                        base = '.'.join(prefix + ([node.module] if node.module else []))
                    else:
                        base = node.module or ''
                    for alias in node.names:
                        if alias.name != '*':
                            bind(alias.asname or alias.name, f'{base}.{alias.name}', node)
                elif isinstance(node, ast.ClassDef):
                    qualified = f'{source.module}.{node.name}'
                    bind(node.name, qualified, node)
                    self.classes[qualified] = ClassInfo(qualified, source, node)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    names = [node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
                    for name in names:
                        bind(name, '', node)
                else:
                    # A conditional import/class/assignment is not an unconditional binding.
                    for child in ast.walk(node):
                        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                            bindings[child.id] = None
                        elif isinstance(child, (ast.Import, ast.ImportFrom)):
                            for alias in child.names:
                                bindings[alias.asname or alias.name.split('.')[0]] = None
                        elif isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            bindings[child.name] = None

    def _resolve_name(self, module: str, expression: ast.AST, trail: list) -> str:
        name = dotted(expression)
        if not name:
            raise TraceUnavailable("dynamic_name")
        first, *rest = name.split('.')
        binding = self.bindings[module].get(first)
        if binding is None or not binding[0]:
            raise TraceUnavailable("unresolved_or_rebound_name")
        target, node = binding
        source = self.sources[module]
        if not isinstance(node, ast.ClassDef):
            trail.append({"kind": "import_binding", "path": source.path, "module": module, "line": node.lineno,
                          "text": ast.get_source_segment(source.text, node), "target": target})
        return self._resolve_qualified('.'.join([target, *rest]), trail, set())

    def _resolve_qualified(self, name: str, trail: list, seen: set) -> str:
        if name in seen:
            raise TraceUnavailable("cyclic_export")
        seen.add(name)
        if name in self.classes:
            info = self.classes[name]
            if info.source.module in self.opaque_modules:
                raise TraceUnavailable("wildcard_import_in_module")
            binding = self.bindings[info.source.module].get(info.node.name)
            if binding is None or binding[0] != name:
                raise TraceUnavailable("ambiguous_class_binding")
            if info.node.decorator_list or info.node.keywords:
                raise TraceUnavailable("decorated_or_metaclass_class")
            return name
        module, _, member = name.rpartition('.')
        if module not in self.sources:
            raise TraceUnavailable("dependency_module_missing")
        if module in self.opaque_modules:
            raise TraceUnavailable("wildcard_import_in_module")
        binding = self.bindings[module].get(member)
        if binding is None or not binding[0]:
            raise TraceUnavailable("export_missing_or_ambiguous")
        target, node = binding
        source = self.sources[module]
        trail.append({"kind": "export_binding", "module": module, "path": source.path, "line": node.lineno,
                      "text": ast.get_source_segment(source.text, node), "target": target})
        return self._resolve_qualified(target, trail, seen)

    def _mro(self, name: str, trail: list) -> list[str]:
        chain = []
        while name:
            if name in chain or len(chain) >= 16:
                raise TraceUnavailable("inheritance_cycle_or_depth")
            name = self._resolve_qualified(name, trail, set())
            chain.append(name)
            info = self.classes[name]
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in {"__new__", "__getattribute__", "__getattr__"} for n in info.node.body):
                raise TraceUnavailable("custom_object_dispatch")
            if not info.node.bases or (len(info.node.bases) == 1 and dotted(info.node.bases[0]) == 'object' and 'object' not in self.bindings[info.source.module]):
                break
            if len(info.node.bases) != 1:
                raise TraceUnavailable("multiple_inheritance")
            base = self._resolve_name(info.source.module, info.node.bases[0], trail)
            trail.append(mark(info, info.node.bases[0], "inherits", target=base))
            name = base
        return chain

    def _method(self, chain: list[str], method: str) -> tuple[ClassInfo, ast.FunctionDef]:
        for name in chain:
            info = self.classes[name]
            nodes = [node for node in info.node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method]
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method
                   for stmt in info.node.body if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for n in ast.walk(stmt)):
                raise TraceUnavailable("conditional_method")
            if len(nodes) > 1:
                raise TraceUnavailable("duplicate_method")
            # Other bindings to the method cannot be resolved safely.
            if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == method for stmt in info.node.body if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(stmt)):
                raise TraceUnavailable("rebound_method")
            if nodes:
                node = nodes[0]
                if node.decorator_list or isinstance(node, ast.AsyncFunctionDef):
                    raise TraceUnavailable("decorated_or_async_constructor_factory")
                return info, node
        raise TraceUnavailable("method_missing")

    def resolve_default(self, class_name: str, parameter: str) -> dict:
        trail = []
        try:
            name = self._resolve_qualified(class_name, trail, set())
            value, provider = self._trace(name, parameter, trail, set())
            return {"status": "resolved", "class": class_name, "parameter": parameter, "value": value,
                    "value_type": type(value).__name__, "provider_class": provider, "trace": trail,
                    "assumptions": ["explicit snapshot module mapping", "no runtime monkeypatching or dynamic import side effects", "parameter omitted by caller"]}
        except TraceUnavailable as exc:
            return {"status": "abstain", "class": class_name, "parameter": parameter, "reason": str(exc), "trace": trail}

    def _trace(self, actual: str, parameter: str, trail: list, seen: set,
               start_after: str | None = None, via_keyword: bool = False) -> tuple[Any, str]:
        key = (actual, parameter, start_after, via_keyword)
        if key in seen or len(seen) >= 12:
            raise TraceUnavailable("forwarding_cycle_or_depth")
        seen = seen | {key}
        chain = self._mro(actual, trail)
        search_chain = chain[chain.index(start_after)+1:] if start_after else chain
        info, constructor = self._method(search_chain, '__init__')
        positional = [*constructor.args.posonlyargs, *constructor.args.args]
        defaults = [None] * (len(positional) - len(constructor.args.defaults)) + list(constructor.args.defaults)
        parameters = {arg.arg: default for arg, default in zip(positional, defaults)}
        parameters.update({arg.arg: default for arg, default in zip(constructor.args.kwonlyargs, constructor.args.kw_defaults)})
        if parameter in parameters:
            if via_keyword and parameter in {arg.arg for arg in constructor.args.posonlyargs}:
                raise TraceUnavailable("positional_only_not_bound_by_kwargs")
            node = parameters[parameter]
            if node is None:
                raise TraceUnavailable("required_parameter_has_no_default")
            value = literal(node)
            trail.append(mark(info, node, "declared_default", parameter=parameter, value=value))
            return value, actual
        if not constructor.args.kwarg:
            raise TraceUnavailable("parameter_not_accepted")
        aliases = {constructor.args.kwarg.arg}
        alias_trace = []
        found = []
        written_attributes = set()

        def key_of(node):
            try:
                return literal(node)
            except TraceUnavailable:
                return None

        def affects_parameter(nodes):
            for stmt in nodes:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                        if isinstance(node.ctx, (ast.Store, ast.Del)) and key_of(node.slice) in (None, parameter):
                            return True
                    if isinstance(node, ast.Name) and node.id in aliases and isinstance(node.ctx, ast.Store):
                        return True
                if escaped_reference(stmt):
                    return True
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases:
                            if node.func.attr in ('get', '__contains__'):
                                continue
                            if node.func.attr == 'pop' and node.args and key_of(node.args[0]) not in (None, parameter):
                                continue
                            return True
                        if any(isinstance(arg, ast.Name) and arg.id in aliases for arg in node.args):
                            return True
                        if any(kw.arg is not None and isinstance(kw.value, ast.Name) and kw.value.id in aliases for kw in node.keywords):
                            return True
                        if any(kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id in aliases for kw in node.keywords):
                            return True
            return False

        def escaped_reference(stmt):
            parents = {child: parent for parent in ast.walk(stmt) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Name) or node.id not in aliases or not isinstance(node.ctx, ast.Load):
                    continue
                parent = parents.get(node)
                if isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is node:
                    continue
                if isinstance(parent, ast.keyword) and parent.arg is None:
                    continue
                return True
            return False

        for stmt in constructor.body:
            if isinstance(stmt, (ast.If, ast.Try, ast.For, ast.While, ast.With, ast.Match)):
                if affects_parameter([stmt]):
                    raise TraceUnavailable("conditional_or_complex_kwargs_flow")
                # Branches may exit before delegation; conservatively refuse.
                if any(isinstance(node, (ast.Return, ast.Raise)) for node in ast.walk(stmt)):
                    raise TraceUnavailable("conditional_exit_before_forwarding")
                continue
            if isinstance(stmt, (ast.Return, ast.Raise)):
                if not found:
                    raise TraceUnavailable("exit_before_forwarding")
                break
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Name) and stmt.value.id in aliases:
                if not all(isinstance(target, ast.Name) for target in stmt.targets):
                    raise TraceUnavailable("kwargs_escape")
                aliases.update(target.id for target in stmt.targets)
                alias_trace.append(mark(info, stmt, "kwargs_alias"))
                continue
            if escaped_reference(stmt):
                raise TraceUnavailable("kwargs_escape")
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                targets = stmt.targets if isinstance(stmt, (ast.Assign, ast.Delete)) else [stmt.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and dotted(target.value) == 'self':
                        written_attributes.add(target.attr)
                    if isinstance(target, ast.Name) and target.id in aliases:
                        raise TraceUnavailable("kwargs_alias_rebound")
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in aliases:
                        key_name = key_of(target.slice)
                        if key_name is None or key_name == parameter:
                            raise TraceUnavailable("forwarded_parameter_modified")
                        alias_trace.append(mark(info, target, "other_parameter_write", key=key_name))
            # Nested lambdas/functions are not executed; any capture of kwargs is unsupported.
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(isinstance(node, ast.Name) and node.id in aliases for node in ast.walk(stmt)):
                    raise TraceUnavailable("kwargs_capture")
                continue
            for call in [node for node in ast.walk(stmt) if isinstance(node, ast.Call)]:
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id in aliases:
                    if call.func.attr == 'pop' and call.args and key_of(call.args[0]) not in (None, parameter):
                        alias_trace.append(mark(info, call, "other_parameter_pop", key=key_of(call.args[0])))
                        continue
                    if call.func.attr in ('get', '__contains__'):
                        continue
                    raise TraceUnavailable("unknown_kwargs_mutation")
                expanded = [kw for kw in call.keywords if kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id in aliases]
                if not expanded:
                    if any(isinstance(arg, ast.Name) and arg.id in aliases for arg in call.args) or any(kw.arg is not None and isinstance(kw.value, ast.Name) and kw.value.id in aliases for kw in call.keywords):
                        raise TraceUnavailable("kwargs_escape")
                    continue
                if len(expanded) != 1 or any(kw.arg is None and kw not in expanded for kw in call.keywords):
                    raise TraceUnavailable("multiple_kwargs_sources")
                if call.args or any(kw.arg == parameter for kw in call.keywords):
                    raise TraceUnavailable("explicit_override_or_positional_forwarding")
                branch = list(alias_trace)
                is_super = (isinstance(call.func, ast.Attribute) and call.func.attr == '__init__' and
                            isinstance(call.func.value, ast.Call) and dotted(call.func.value.func) == 'super' and
                            not call.func.value.args and not call.func.value.keywords)
                if is_super:
                    if 'super' in self.bindings[info.source.module]:
                        raise TraceUnavailable("shadowed_super")
                    branch.append(mark(info, call, "super_forward", parameter=parameter))
                    value, provider = self._trace(actual, parameter, branch, seen, start_after=info.qualified, via_keyword=True)
                else:
                    target_expression = call.func
                    target_module = info.source.module
                    if isinstance(call.func, ast.Call):
                        factory_call = call.func
                        if not isinstance(factory_call.func, ast.Attribute) or dotted(factory_call.func.value) != 'self' or factory_call.args or factory_call.keywords:
                            raise TraceUnavailable("dynamic_factory")
                        if factory_call.func.attr in written_attributes or '__class__' in written_attributes:
                            raise TraceUnavailable("factory_rebound")
                        factory_info, factory = self._method(chain, factory_call.func.attr)
                        required_positional = len(factory.args.posonlyargs) + len(factory.args.args) - len(factory.args.defaults)
                        if required_positional > 1 or any(value is None for value in factory.args.kw_defaults):
                            raise TraceUnavailable("factory_requires_arguments")
                        body = [node for node in factory.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
                        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
                            raise TraceUnavailable("dynamic_factory")
                        branch.append(mark(factory_info, body[0], "factory_target"))
                        target_expression, target_module = body[0].value, factory_info.source.module
                    target = self._resolve_name(target_module, target_expression, branch)
                    branch.append(mark(info, call, "kwargs_forward", target=target, parameter=parameter))
                    value, provider = self._trace(target, parameter, branch, seen, via_keyword=True)
                found.append((value, provider, branch))
        if len(found) != 1:
            raise TraceUnavailable("no_unique_forwarding_target")
        value, provider, branch = found[0]
        trail.extend(branch)
        return value, provider
