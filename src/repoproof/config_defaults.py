"""Bounded AST tracing of fields of a declared initial configuration object.

This resolves a constructor parameter's *declared default object*, not the
runtime value after the consumer's body, explicit overrides or later mutation.
Target source is parsed only. No imports, calls, annotations or decorators run.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from .contract_graph import Source, TraceUnavailable, dotted, literal


@dataclass(frozen=True)
class _Value:
    source: Source
    node: ast.AST
    identity: str | None


def _mark(source, node, kind, **extra):
    return {'kind':kind, 'module':source.module, 'path':source.path,
            'line':node.lineno, 'end_line':getattr(node,'end_lineno',node.lineno),
            'text':ast.get_source_segment(source.text,node) or '',
            'source_line':source.text.split('\n')[node.lineno-1], **extra}


class ConfigDefaults:
    """Follow unique bindings to simple, non-inherited config constructors."""

    def __init__(self, sources: list[Source]):
        self.sources={source.module:source for source in sources}
        if len(self.sources)!=len(sources):
            raise ValueError('Duplicate module mapping')
        self.trees={}
        self.bindings={}
        self.opaque=set()
        for source in sources:
            tree=ast.parse(source.text,filename=source.path)
            self.trees[source.module]=tree
            bindings=self.bindings[source.module]={}
            def bind(name,kind,node,target=None):
                bindings[name]=None if name in bindings else (kind,node,target)
            for node in tree.body:
                if isinstance(node,ast.Import):
                    for alias in node.names:
                        bind(alias.asname or alias.name.split('.')[0],'import',node,
                             alias.name if alias.asname else alias.name.split('.')[0])
                elif isinstance(node,ast.ImportFrom):
                    if any(alias.name=='*' for alias in node.names):
                        self.opaque.add(source.module)
                    package=source.module.split('.') if source.is_package else source.module.split('.')[:-1]
                    if node.level>len(package):
                        self.opaque.add(source.module)
                        continue
                    prefix=package[:len(package)-node.level+1] if node.level else []
                    base='.'.join(prefix+([node.module] if node.module else []))
                    for alias in node.names:
                        if alias.name!='*':bind(alias.asname or alias.name,'import',node,f'{base}.{alias.name}')
                elif isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    bind(node.name,'definition',node)
                elif isinstance(node,(ast.Assign,ast.AnnAssign)):
                    targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target,ast.Name) and len(targets)==1 and node.value is not None:
                            bind(target.id,'value',node,node.value)
                        else:
                            for child in ast.walk(target):
                                if isinstance(child,ast.Name) and isinstance(child.ctx,ast.Store):
                                    bind(child.id,'unsupported',node)
                else:
                    # Conditional definitions, deletion and augmentation are not
                    # unique unconditional bindings. Do not choose a branch.
                    for child in ast.walk(node):
                        if isinstance(child,ast.Name) and isinstance(child.ctx,(ast.Store,ast.Del)):
                            bindings[child.id]=None
                        elif isinstance(child,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                            bindings[child.name]=None
                        elif isinstance(child,(ast.Import,ast.ImportFrom)):
                            for alias in child.names:
                                bindings[alias.asname or alias.name.split('.')[0]]=None
                        elif isinstance(child,(ast.MatchAs,ast.MatchStar,ast.ExceptHandler)) and child.name:
                            bindings[child.name]=None
                        elif isinstance(child,ast.MatchMapping) and child.rest:
                            bindings[child.rest]=None
            # A function may explicitly rebind a module global. Its execution
            # is unknown, so even a unique top-level declaration is insufficient.
            for scope in ast.walk(tree):
                if isinstance(scope,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    globals_seen={name for child in ast.walk(scope) if isinstance(child,ast.Global)
                                  for name in child.names}
                    for child in ast.walk(scope):
                        if (isinstance(child,ast.Name) and isinstance(child.ctx,(ast.Store,ast.Del))
                                and child.id in globals_seen):
                            bindings[child.id]=None
                        elif (isinstance(child,(ast.MatchAs,ast.MatchStar,ast.ExceptHandler))
                              and child.name in globals_seen):
                            bindings[child.name]=None
                        elif isinstance(child,ast.MatchMapping) and child.rest in globals_seen:
                            bindings[child.rest]=None

    def _qualified(self, name, trace, seen):
        modules=[module for module in self.sources if name.startswith(module+'.')]
        if not modules:
            raise TraceUnavailable('dependency_module_missing')
        module=max(modules,key=len)
        member=name[len(module)+1:]
        if '.' in member:
            raise TraceUnavailable('nested_or_dynamic_member')
        return self._binding(module,member,trace,seen)

    def _binding(self,module,name,trace,seen):
        key=f'{module}.{name}'
        if key in seen or len(seen)>=24:
            raise TraceUnavailable('binding_cycle_or_depth')
        seen=seen|{key}
        if module in self.opaque:
            raise TraceUnavailable('wildcard_or_invalid_import')
        binding=self.bindings.get(module,{}).get(name)
        if binding is None:
            raise TraceUnavailable('missing_ambiguous_or_rebound_binding')
        kind,node,target=binding
        source=self.sources[module]
        if kind=='import':
            trace.append(_mark(source,node,'import_binding',target=target))
            return self._qualified(target,trace,seen)
        if kind=='definition':
            return _Value(source,node,key)
        if kind=='value':
            trace.append(_mark(source,node,'configuration_binding',binding=key))
            if isinstance(target,(ast.Name,ast.Attribute)):
                return self._expression(source,target,trace,seen)
            return _Value(source,target,key)
        raise TraceUnavailable('unsupported_binding')

    def _expression(self,source,node,trace,seen):
        if isinstance(node,ast.Name):
            return self._binding(source.module,node.id,trace,seen)
        if isinstance(node,ast.Attribute):
            name=dotted(node)
            if not name:
                raise TraceUnavailable('dynamic_attribute')
            first,*rest=name.split('.')
            binding=self.bindings[source.module].get(first)
            if not binding or binding[0]!='import':
                raise TraceUnavailable('attribute_not_explicit_module_import')
            _,import_node,target=binding
            trace.append(_mark(source,import_node,'import_binding',target=target))
            return self._qualified('.'.join([target,*rest]),trace,seen)
        return _Value(source,node,None)

    @staticmethod
    def _method(info,name):
        node=info.node
        if not isinstance(node,ast.ClassDef):
            raise TraceUnavailable('target_not_class')
        if node.decorator_list or node.keywords:
            raise TraceUnavailable('decorated_or_metaclass_class')
        definitions=[statement for statement in node.body
                     if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef)) and statement.name==name]
        others=[statement for statement in node.body
                if not (isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef)) and statement.name==name)]
        if len(definitions)!=1:
            raise TraceUnavailable('missing_or_duplicate_constructor')
        for statement in others:
            for child in ast.walk(statement):
                if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and child.name==name:
                    raise TraceUnavailable('conditional_constructor')
                if isinstance(child,ast.Name) and isinstance(child.ctx,(ast.Store,ast.Del)) and child.id==name:
                    raise TraceUnavailable('rebound_constructor')
        method=definitions[0]
        if isinstance(method,ast.AsyncFunctionDef) or method.decorator_list:
            raise TraceUnavailable('decorated_or_async_constructor')
        return method

    @staticmethod
    def _parameters(method):
        positional=[*method.args.posonlyargs,*method.args.args]
        defaults=[None]*(len(positional)-len(method.args.defaults))+list(method.args.defaults)
        return {arg.arg:default for arg,default in [*zip(positional,defaults),*zip(method.args.kwonlyargs,method.args.kw_defaults)]}

    @staticmethod
    def _reject_class_scope_bindings(info,expression):
        """Method defaults execute in the class namespace, not the module.

        Class namespaces and branch/order-sensitive bindings are not modeled.
        Reject possible shadowing instead of accidentally using a module value.
        Function bodies and nested class bodies do not bind this namespace.
        """
        referenced={node.id for node in ast.walk(expression)
                    if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load)}
        if not referenced:return
        bound=set()
        opaque=False

        def visit(node):
            nonlocal opaque
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                bound.add(node.name)
                # Defaults, decorators, annotations and bases execute while
                # defining the member; its body belongs to a different scope.
                for attr,value in ast.iter_fields(node):
                    if attr=='body':continue
                    for child in value if isinstance(value,list) else [value]:
                        if isinstance(child,ast.AST):visit(child)
                return
            if isinstance(node,ast.Name) and isinstance(node.ctx,(ast.Store,ast.Del)):
                bound.add(node.id)
            elif isinstance(node,(ast.Import,ast.ImportFrom)):
                for alias in node.names:
                    if alias.name=='*':opaque=True
                    else:bound.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node,ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node,(ast.MatchAs,ast.MatchStar)) and node.name:
                bound.add(node.name)
            elif isinstance(node,ast.MatchMapping) and node.rest:
                bound.add(node.rest)
            for child in ast.iter_child_nodes(node):visit(child)

        for statement in info.node.body:visit(statement)
        if opaque or referenced & bound:
            raise TraceUnavailable('class_scope_binding_not_modeled')

    def _check_observed_mutations(self,identities):
        identities={identity for identity in identities if identity is not None}
        if not identities:return
        for module,tree in self.trees.items():
            source=self.sources[module]
            for node in ast.walk(tree):
                if isinstance(node,(ast.Attribute,ast.Subscript)) and isinstance(node.ctx,(ast.Store,ast.Del)):
                    expression=node.value
                    while isinstance(expression,(ast.Name,ast.Attribute)):
                        try:
                            value=self._expression(source,expression,[],set())
                            if value.identity in identities:
                                raise TraceUnavailable('observed_configuration_or_class_mutation')
                        except TraceUnavailable as error:
                            if str(error)=='observed_configuration_or_class_mutation':raise
                        expression=expression.value if isinstance(expression,ast.Attribute) else None
                # Passing the actual bound object/class to an unknown call may
                # mutate it. Reading one of its immutable fields is different.
                if isinstance(node,ast.Call):
                    if isinstance(node.func,ast.Attribute):
                        receiver=node.func.value
                        while isinstance(receiver,(ast.Name,ast.Attribute)):
                            try:value=self._expression(source,receiver,[],set())
                            except TraceUnavailable:value=None
                            if value is not None and value.identity in identities:
                                raise TraceUnavailable('configuration_receiver_call')
                            receiver=receiver.value if isinstance(receiver,ast.Attribute) else None
                    for argument in [*node.args,*[kw.value for kw in node.keywords]]:
                        try:value=self._expression(source,argument,[],set())
                        except TraceUnavailable:continue
                        if value.identity in identities:
                            raise TraceUnavailable('configuration_escapes_to_call')

    def resolve_constructor_field(self,class_name: str,parameter: str,field: str) -> dict:
        trace=[]
        result={'status':'abstain','class':class_name,'parameter':parameter,'field':field,'trace':trace,
                'scope':'initial field of the declared default configuration object; not consumer runtime behavior',
                'target_code_executed':False,
                'assumptions':['Explicit source snapshot mapping; dependencies are not imported.',
                               'Caller omits the configuration parameter; explicit overrides and consumer-body branches are not evaluated.',
                               'No unmodeled monkeypatching, descriptor effects, import-time side effects or mutations through unmodeled aliases.']}
        try:
            consumer=self._qualified(class_name,trace,set())
            method=self._method(consumer,'__init__')
            params=self._parameters(method)
            if parameter not in params or params[parameter] is None:
                raise TraceUnavailable('configuration_parameter_has_no_declared_default')
            default=params[parameter]
            trace.append(_mark(consumer.source,default,'consumer_default',parameter=parameter))
            self._reject_class_scope_bindings(consumer,default)
            object_value=self._expression(consumer.source,default,trace,set())
            call=object_value.node
            if not isinstance(call,ast.Call):
                raise TraceUnavailable('declared_default_is_not_configuration_construction')
            if any(isinstance(arg,ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
                raise TraceUnavailable('expanded_constructor_arguments')
            config=self._expression(object_value.source,call.func,trace,set())
            initializer=self._method(config,'__init__')
            for constructor_default in self._parameters(initializer).values():
                if constructor_default is not None:
                    self._reject_class_scope_bindings(config,constructor_default)
            config_class=config.node
            if config_class.bases and not (len(config_class.bases)==1 and dotted(config_class.bases[0])=='object'
                                           and 'object' not in self.bindings[config.source.module]):
                raise TraceUnavailable('inherited_configuration_class')
            if initializer.args.vararg or initializer.args.kwarg:
                raise TraceUnavailable('variadic_configuration_constructor')
            forbidden={'__new__','__setattr__','__getattribute__','__getattr__','__delattr__',field}
            for statement in config_class.body:
                if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    if statement.name in forbidden or statement.decorator_list:
                        raise TraceUnavailable('custom_object_or_field_dispatch')
                elif isinstance(statement,ast.Pass):continue
                elif isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Constant) and isinstance(statement.value.value,str):continue
                else:
                    raise TraceUnavailable('unsupported_configuration_class_body')
            self._check_observed_mutations({object_value.identity,config.identity})
            positional=[*initializer.args.posonlyargs,*initializer.args.args]
            if not positional:
                raise TraceUnavailable('constructor_instance_parameter_missing')
            instance=positional[0].arg
            names=[arg.arg for arg in positional[1:]]
            if len(call.args)>len(names):raise TraceUnavailable('too_many_constructor_arguments')
            arguments=dict(zip(names,call.args))
            posonly={arg.arg for arg in initializer.args.posonlyargs[1:]}
            for keyword in call.keywords:
                if keyword.arg in arguments or keyword.arg in posonly:
                    raise TraceUnavailable('duplicate_or_positional_only_argument')
                arguments[keyword.arg]=keyword.value
            defaults=self._parameters(initializer)
            defaults.pop(instance,None)
            if not set(arguments)<=set(defaults):raise TraceUnavailable('unknown_constructor_argument')
            values={}
            for name,default_node in defaults.items():
                expr=arguments.get(name,default_node)
                if expr is None:raise TraceUnavailable('missing_required_constructor_argument')
                values[name]=literal(expr)
            assignments={}
            for statement in initializer.body:
                if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Constant) and isinstance(statement.value.value,str):continue
                if isinstance(statement,ast.Pass):continue
                if isinstance(statement,ast.Assign) and len(statement.targets)==1:
                    target,value=statement.targets[0],statement.value
                elif isinstance(statement,ast.AnnAssign) and statement.value is not None:
                    target,value=statement.target,statement.value
                else:raise TraceUnavailable('constructor_not_pure_field_assignment')
                if (not isinstance(target,ast.Attribute) or not isinstance(target.value,ast.Name)
                        or target.value.id!=instance or not isinstance(value,ast.Name) or value.id not in values):
                    raise TraceUnavailable('constructor_field_not_direct_parameter')
                if target.attr in assignments:raise TraceUnavailable('reassigned_configuration_field')
                if target.attr in {'__dict__','__class__'}:raise TraceUnavailable('custom_instance_storage')
                assignments[target.attr]=(value.id,statement)
            if field not in assignments:raise TraceUnavailable('configuration_field_not_assigned')
            selected,assignment=assignments[field]
            selected_expr=arguments.get(selected,defaults[selected])
            explicit=selected in arguments
            trace.append(_mark(object_value.source,call,'configuration_construction',configuration_class=config.identity))
            trace.append(_mark(object_value.source if explicit else config.source,selected_expr,
                               'explicit_configuration_argument' if explicit else 'configuration_constructor_default',
                               argument=selected,value=values[selected]))
            trace.append(_mark(config.source,assignment,'configuration_field_assignment',field=field,argument=selected))
            result.update(status='resolved',reason='initial_configuration_field_resolved',
                          value=values[selected],value_type=type(values[selected]).__name__,
                          configuration_class=config.identity,configuration_binding=object_value.identity)
        except TraceUnavailable as error:
            result['reason']=str(error)
        return result
