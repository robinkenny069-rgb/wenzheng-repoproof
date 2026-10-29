"""Audit explicit NumPy-style parameter defaults against lexical declarations.

Only literal header claims are supported. No imports, decorator execution,
template expansion, inherited API reconstruction or runtime behavior proof.
Every recognized parameter field and every discovered function is retained.
"""
from __future__ import annotations
import ast
from collections import Counter
from decimal import Decimal
import io
import math
import re
import tokenize

_HEADER = re.compile(r'^(?P<names>\*{0,2}[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)\s*:\s*(?P<type>.+)$')
_DEFAULT = re.compile(r'\bdefault\s*[:=]\s*(.+)$', re.I)


def scalar(expression):
    try:
        # literal_eval ignores trailing Python comments. In prose those may
        # qualify a value, so do not silently discard them. Quoted '#' is fine.
        lexemes = tokenize.generate_tokens(io.StringIO(expression).readline)
        if any(token.type == tokenize.COMMENT for token in lexemes):
            return {'status':'unsupported', 'reason':'non_literal_or_qualified_default'}
        value = ast.literal_eval(expression)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError, tokenize.TokenError):
        return {'status':'unsupported', 'reason':'non_literal_or_qualified_default'}
    if value is not None and type(value) not in (bool, int, float, str):
        return {'status':'unsupported', 'reason':'non_scalar_default'}
    if type(value) is float and not math.isfinite(value):
        return {'status':'unsupported', 'reason':'non_finite_default'}
    return {'status':'parsed', 'value':value, 'value_type':type(value).__name__}


def declared_parameters(source, node):
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [None]*(len(positional)-len(node.args.defaults))+list(node.args.defaults)
    pairs = list(zip(positional,defaults))+list(zip(node.args.kwonlyargs,node.args.kw_defaults))
    result = {}
    for arg, default in pairs:
        if default is None:
            result[arg.arg] = {'status':'unsupported','reason':'required_parameter_has_no_default','line':arg.lineno}
        else:
            expression = ast.get_source_segment(source,default) or ''
            result[arg.arg] = {**scalar(expression),'expression':expression,'line':default.lineno,
                               'source_line':source.split('\n')[default.lineno-1]}
    return result


def numpy_fields(source, node, symbol, path):
    if not node.body or not isinstance(node.body[0],ast.Expr) or not isinstance(node.body[0].value,ast.Constant) or not isinstance(node.body[0].value.value,str):
        return [], 'no_literal_docstring'
    literal = node.body[0].value
    raw = ast.get_source_segment(source,literal) or ''
    tokens = list(tokenize.generate_tokens(io.StringIO(raw).readline))
    if (sum(token.type==tokenize.STRING for token in tokens)!=1
            or literal.value.count('\n')!=raw.count('\n')
            or any(separator in literal.value for separator in ('\r','\v','\f','\x85','\u2028','\u2029'))):
        return [], 'docstring_source_mapping_unsupported'
    lines = literal.value.split('\n')
    indentation = lambda line: len(line)-len(line.lstrip())
    # Like a normal dedent, ignore the first line following the opening quotes.
    # Nested RST literal examples must not create real parameter sections.
    body_indent = min((indentation(line) for line in lines[1:] if line.strip()), default=0)
    headings = {}
    for i in range(len(lines)-1):
        if (lines[i].strip() and re.fullmatch(r'-{3,}',lines[i+1].strip())
                and (i == 0 or indentation(lines[i]) == body_indent)
                and indentation(lines[i+1]) == body_indent):
            headings[i] = lines[i].strip()
    fields = []
    uncertain = False
    found_section = False
    for start,title in headings.items():
        if title not in ('Parameters','Other Parameters'):
            continue
        found_section = True
        end = next((i for i in headings if i>start),len(lines))
        indent = body_indent
        current = None
        for i in range(start+2,end):
            line = lines[i]
            at_indent = len(line)-len(line.lstrip())==indent
            match = _HEADER.fullmatch(line.strip()) if at_indent else None
            if match:
                if current: fields.append(current)
                current = {'block_id':f'{path}:{symbol}:{literal.lineno+i}', 'symbol':symbol,'path':path,
                           'parameters':[name.strip() for name in match['names'].split(',')],
                           'header':line.strip(), 'descriptor':match['type'], 'start_line':literal.lineno+i,
                           'end_line':literal.lineno+i, 'text':line.strip()}
            elif current:
                current['text'] += '\n'+line.rstrip()
                current['end_line'] = literal.lineno+i
            elif line.strip():
                uncertain = True
        if current: fields.append(current)
    for field in fields:
        field['text'] = field['text'].rstrip()
        field['end_line'] = field['start_line']+field['text'].count('\n')
    status = 'parameter_sections_parsed' if found_section else 'no_numpy_parameter_section'
    if uncertain or re.search(r'%\([^)]+\)s',literal.value):
        status = 'partial_or_templated_parameter_sections'
    return fields, status


def header_claim(field):
    mentions = list(re.finditer(r'\bdefaults?\b',field['text'],re.I))
    if len(mentions)!=1:
        return {'status':'unsupported','reason':'no_unique_explicit_header_default'}
    matches = list(_DEFAULT.finditer(field['descriptor']))
    if len(matches)!=1:
        return {'status':'unsupported','reason':'default_not_in_supported_header'}
    prefix = field['descriptor'][:matches[0].start()]
    if re.search(r'\b(?:not|never|if|when|previous|previously|former|formerly|old|future)\b',prefix,re.I):
        return {'status':'unsupported','reason':'qualified_or_historical_header'}
    expression = matches[0].group(1).strip()
    if expression.startswith('``') and expression.endswith('``'):
        expression = expression[2:-2]
    elif expression.startswith('`') and expression.endswith('`'):
        expression = expression[1:-1]
    # Parse the complete expression: units, qualifiers and alternatives abstain.
    return {**scalar(expression),'expression':expression,'line':field['start_line']}


def compare_field(field, parameter, declaration, *, ambiguous=False):
    claim = header_claim(field)
    result = {'symbol':field['symbol'],'parameter':parameter,'block_id':field['block_id'],'document':field,
              'claim':claim,'declaration':declaration,'status':'abstain','automatic_edit':False,
              'scope':'literal documented default versus lexical declaration; not runtime behavior'}
    if ambiguous:
        result['reason']='ambiguous_definition_or_documentation'
    elif declaration is None:
        result['reason']='parameter_not_explicit_in_signature'
    elif declaration['status']!='parsed':
        result['reason']=declaration['reason']
    elif claim['status']!='parsed':
        result['reason']=claim['reason']
    elif declaration['value'] is None and claim['value'] is not None:
        result['reason']='sentinel_requires_behavior_review'
    else:
        a,b = declaration['value'],claim['value']
        numeric = type(a) in (int,float) and type(b) in (int,float)
        equal = Decimal(str(a))==Decimal(str(b)) if numeric else type(a) is type(b) and a==b
        result.update(status='consistent_declared_value' if equal else 'declared_default_mismatch_candidate',
                      reason='literal_values_agree' if equal else 'literal_values_differ',
                      comparison='numeric_value' if numeric else 'type_and_value')
    return result


def audit_source(source, path='source.py'):
    tree = ast.parse(source, filename=path)
    functions = []
    def visit(statements, prefix='', conditional=False):
        for node in statements:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                symbol = prefix+node.name
                fields,doc_status = numpy_fields(source,node,symbol,path)
                functions.append({'symbol':symbol,'path':path,'line':node.lineno,'conditional':conditional,
                                  'decorated':bool(node.decorator_list),'parameters':declared_parameters(source,node),
                                  'documentation_status':doc_status,'fields':fields})
                visit(node.body,symbol+'.<locals>.',conditional)
            elif isinstance(node,ast.ClassDef):
                visit(node.body,prefix+node.name+'.',conditional)
            else:
                # Conditional definitions are inventoried but not treated as unique APIs.
                for child in ast.iter_child_nodes(node):
                    if isinstance(child,ast.stmt): visit([child],prefix,True)
                    elif isinstance(child,(ast.ExceptHandler,ast.match_case)): visit(child.body,prefix,True)
    visit(tree.body)
    symbol_counts = Counter(f['symbol'] for f in functions)
    rows = []
    blocks = []
    for function in functions:
        fields = function.pop('fields')
        blocks.extend(fields)
        name_counts = Counter(name for field in fields for name in field['parameters'])
        for field in fields:
            for parameter in field['parameters']:
                row = compare_field(field,parameter,function['parameters'].get(parameter),
                                    ambiguous=function['conditional'] or symbol_counts[function['symbol']]>1 or name_counts[parameter]>1)
                row['decorated_declaration'] = function['decorated']
                rows.append(row)
    return {'path':path,'functions':functions,'blocks':blocks,'rows':rows,
            'counts':dict(Counter(row['status'] for row in rows)),
            'inventory':{'function_definitions':len(functions),'parameter_blocks':len(blocks),'parameter_rows':len(rows),
                         'documentation_status':dict(Counter(f['documentation_status'] for f in functions))},
            'code_executed':False,'template_expansion':False}
