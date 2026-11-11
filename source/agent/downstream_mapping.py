"""Lossless projection from training semantics to downstream navigation classes."""
from __future__ import annotations

import copy


TRAVERSABLE_AREA_MAPPING = {
    "target_label": "traversable_area",
    "operation": "union",
    "source_labels": ["floor", "slope"],
    "source_layer": "training_annotation",
    "target_layer": "downstream_navigation",
    "preserve_source_labels": True,
}


def mapping_contract():
    return copy.deepcopy(TRAVERSABLE_AREA_MAPPING)


def project_shapes(shapes, *, source_layer):
    """Project floor/slope parts without mutating or merging source truth."""
    if source_layer not in {"machine_candidate", "human_truth"}:
        raise ValueError("source_layer must be machine_candidate or human_truth")
    if not isinstance(shapes, list):
        raise ValueError("shapes must be an array")
    traversable, passthrough = [], []
    for shape in shapes:
        if not isinstance(shape, dict) or not isinstance(shape.get("label"), str):
            raise ValueError("every shape must contain a semantic label")
        item = copy.deepcopy(shape)
        if item["label"] in TRAVERSABLE_AREA_MAPPING["source_labels"]:
            item["source_semantic_label"] = item["label"]
            item["label"] = TRAVERSABLE_AREA_MAPPING["target_label"]
            traversable.append(item)
        else:
            passthrough.append(item)
    return {
        "mapping": mapping_contract(),
        "source_layer": source_layer,
        "machine_candidate_only": source_layer == "machine_candidate",
        "training_labels_preserved": True,
        "traversable_area_parts": traversable,
        "passthrough_shapes": passthrough,
    }
