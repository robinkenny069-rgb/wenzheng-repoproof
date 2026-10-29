"""Source-only retrieval text; no labels, values, synonyms or repair content.

This representation is separate from original source evidence and cannot change
an inconsistency verdict. English identifier expansion is a retrieval heuristic.
"""
import re


def identifier_words(text: str) -> str:
    if not isinstance(text,str):
        raise TypeError('Expected text')
    def expand(match):
        token=match.group(0)
        token=re.sub(r'([A-Z]+)([A-Z][a-z])',r'\1 \2',token)
        token=re.sub(r'([a-z0-9])([A-Z])',r'\1 \2',token)
        return re.sub(r'_+',' ',token).strip()
    return re.sub(r'[A-Za-z_][A-Za-z0-9_]*',expand,text)


def focused_query(local_owner: str, parameter: str) -> str:
    """Caller supplies declaration identity only, never default or doc text."""
    if not isinstance(local_owner,str) or not isinstance(parameter,str) or not local_owner.strip() or not parameter.strip():
        raise ValueError('Nonempty source owner and parameter are required')
    owner=identifier_words(local_owner).replace('.',' ')
    field=identifier_words(parameter)
    return ' '.join(f'{owner} {field} default value.'.split())
