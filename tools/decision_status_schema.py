"""Experimental syntax rule: continue requires an operation; done stays allowed."""
from copy import deepcopy


def continue_requires_operation(schema):
    done=deepcopy(schema)
    continued=deepcopy(schema)
    done['properties']['status']['enum']=['done']
    continued['properties']['status']['enum']=['continue']
    continued['properties']['operations']['minItems']=1
    # Keep legacy done-with-edits valid: production still observes those edits.
    return {'anyOf':[done,continued]}
