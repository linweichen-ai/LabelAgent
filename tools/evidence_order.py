"""An isolated prompt experiment; preserves the production schema and tools."""
from contextlib import contextmanager

REASON_FIRST_HINT = (
    '\nSerialization order for this review: emit the existing reason field first, '
    'giving one short sentence of visible evidence, then status, hole_assessment '
    'and operations (and sam_assessment if required). Determine status and '
    'operations from that evidence. Use exactly the existing fields and tool '
    'rules; done and empty operations remain allowed when supported.'
)


@contextmanager
def scoped_evidence_order(policy, on_response):
    if policy not in {'baseline', 'reason_first'}:
        raise ValueError('Unknown evidence order policy')
    from source.agent import workflow
    original = workflow.chat

    def chat(config, instruction, image, extra_images=(), response_schema=None):
        hint = REASON_FIRST_HINT if policy == 'reason_first' else ''
        response = original(config, instruction + hint, image,
                            extra_images=extra_images, response_schema=response_schema)
        keys = list(response) if isinstance(response, dict) else []
        on_response({'keys': keys, 'reason_first_observed': bool(keys and keys[0] == 'reason')})
        return response

    workflow.chat = chat
    try:
        yield
    finally:
        workflow.chat = original
