"""Constrain syntax and allowed IDs without deciding the model's edits."""


def obj(properties, required=None):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required,
            "additionalProperties": False}


def bounded_text(limit):
    # Native JSON string bounds avoid compiling a large finite regex for each
    # request, and retain normal JSON quote/escape handling in the decoder.
    return {"type": "string", "minLength": 1, "maxLength": limit}


def reason(limit=180):
    return bounded_text(limit)


def description_schema(labels):
    prompt = obj({"label": {"type": "string", "enum": labels},
                  "text": bounded_text(40)})
    return obj({"description": bounded_text(700),
                "prompts": {"type": "array", "items": prompt, "minItems": len(labels), "maxItems": len(labels)}})


def visible_label_schema(labels):
    """Strict VLM contract for known-label visibility and open-set hints.

    Assessments are keyed by the exact registry labels.  Required object keys
    make duplicates and omissions structurally impossible, including for a new
    dataset whose registry was not known when this code was written.
    """
    assessment_value = obj({
        "evidence": {"type": "string", "enum": ["present", "absent", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        # Visibility is a registry gate, not a captioning task.  One compact
        # visual clause is enough for audit and keeps local constrained
        # decoding responsive when the registry contains many labels.
        "reason": reason(72),
    })
    unknown = obj({
        "candidate_name": bounded_text(80),
        "evidence": reason(96),
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    })
    return obj({
        "assessments": obj({label: assessment_value for label in labels}),
        "unknown_candidates": {"type": "array", "items": unknown, "maxItems": 5},
    })


def review_schema(labels, shapes, holes, sam_pending=False):
    ids = [shape["id"] for shape in shapes]
    hole_ids = [item["id"] for item in holes]
    identifier = {"type": "integer", "enum": ids}
    one = {"type": "array", "items": identifier, "minItems": 1, "maxItems": 1}
    polygon = {"type": "array", "minItems": 3, "maxItems": 32,
               "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}
    def operation(names, extra=None, shape_ids=one):
        return obj({"operation": {"type": "string", "enum": names}, "shape_ids": shape_ids,
                    **(extra or {}), "reason": reason(120)})
    tools = []
    if ids:
        tools = [operation(["remove"]), operation(["reclassify"], {"label": {"type": "string", "enum": labels}}),
                 operation(["add_region", "erase_region", "split"], {"polygon": polygon}),
                 operation(["sam_points"], {"polygon": polygon, "points": {"type": "array", "minItems": 1, "maxItems": 32,
                     "items": obj({"x": {"type": "number"}, "y": {"type": "number"}, "label": {"type": "integer", "enum": [0, 1]}})}})]
        if len(ids) >= 2:
            tools.append(operation(["merge"], {"label": {"type": "string", "enum": labels}},
                                   {"type": "array", "items": identifier, "minItems": 2, "maxItems": min(32, len(ids))}))
        if hole_ids:
            tools.append(operation(["fill_gap"], {"polygon": polygon, "hole_id": {"type": "string", "enum": hole_ids}}))
    assessments = {"type": "array", "maxItems": len(holes), "minItems": len(holes),
                   "items": obj({"hole_id": {"type": "string", "enum": hole_ids or ["unused"]},
                                 "semantic": {"type": "string", "enum": ["ground", "non_ground"]}, "reason": reason(100)})}
    operations = {"type": "array", "maxItems": 4 if tools else 0,
                  "items": {"anyOf": tools} if tools else {"type": "object"}}
    properties = {"status": {"type": "string", "enum": ["continue", "done"]}, "reason": reason(120),
                  "hole_assessment": assessments, "operations": operations}
    if sam_pending:
        properties["sam_assessment"] = obj({"improved": {"type": "boolean"}, "reason": reason(120)})
    return obj(properties)
