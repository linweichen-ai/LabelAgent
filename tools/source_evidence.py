"""Require physical evidence in the clean view; no image or label rewriting."""
import hashlib
import json
from tools.source_last import source_last_payload as BASE_PAYLOAD

SOURCE_EVIDENCE_INSTRUCTION = (
    'Before deciding the category, compare the same selected physical surface '
    'in both images. The class must be supported by visible material or directly '
    'connected object structure in clean image 2. Do not infer class from the '
    'outline silhouette. If only the colored guides in image 1 make an '
    'object-shaped boundary while image 2 shows a smooth field or lighting '
    'gradient without physical object structure, return uncertain. Describe '
    'structure actually visible in image 2, not a colored guide.'
)


def source_evidence_payload(fixture, assets, case, view, guard=False):
    body, digest = BASE_PAYLOAD(fixture, assets, case, view, guard)
    if guard:
        return body, digest
    result = json.loads(json.dumps(body))
    result['messages'][0]['content'][0]['text'] += '\n\n' + SOURCE_EVIDENCE_INSTRUCTION
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result, digest
