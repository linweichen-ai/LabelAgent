"""Experimental decoding constraint: a class change must change the current label.

Only used inside a standalone comparison process. Does not modify production
workflow files, inject edits, reinterpret done, or decide the correct class.
"""
from contextlib import contextmanager
from copy import deepcopy


def exclude_identity_reclassifications(schema,labels,shapes):
    if len(set(labels))!=len(labels) or not labels:
        raise ValueError('Distinct allowed labels are required')
    ids=[shape.get('id') for shape in shapes]
    if len(set(ids))!=len(ids) or any(type(identifier) is not int for identifier in ids):
        raise ValueError('Candidate IDs must be unique integers')
    if any(shape.get('label') not in labels for shape in shapes):
        raise ValueError('Known current candidate labels are required')
    result=deepcopy(schema)
    operations=result['properties']['operations']
    if 'anyOf' not in operations['items']:
        return result
    variants=[]
    for item in operations['items']['anyOf']:
        if item['properties']['operation'].get('enum')!=['reclassify']:
            variants.append(item)
            continue
        for target in labels:
            eligible=[shape['id'] for shape in shapes if shape['label']!=target]
            if not eligible:
                continue
            branch=deepcopy(item)
            branch['properties']['shape_ids']['items']['enum']=eligible
            branch['properties']['label']['enum']=[target]
            variants.append(branch)
    operations['items']['anyOf']=variants
    return result


@contextmanager
def scoped_schema_policy(policy):
    """Change only schema construction in this isolated, sequential experiment."""
    if policy not in {'baseline','exclude_identity'}:
        raise ValueError('Unknown reclassification policy')
    from source.agent import protocol_schema
    original=protocol_schema.review_schema
    if policy=='baseline':
        yield
        return
    def constrained(labels,shapes,holes,sam_pending=False):
        return exclude_identity_reclassifications(original(labels,shapes,holes,sam_pending),labels,shapes)
    protocol_schema.review_schema=constrained
    try:
        yield
    finally:
        protocol_schema.review_schema=original
