"""Constrain syntax and allowed IDs without deciding the model's edits."""


def obj(properties, required=None):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required,
            "additionalProperties": False}


def reason():
    return {"type": "string", "pattern": "[ -~]{1,180}"}


def description_schema(labels):
    prompt = obj({"label": {"type": "string", "enum": labels},
                  "text": {"type": "string", "pattern": "[A-Za-z0-9 -]{1,40}"}})
    return obj({"description": {"type": "string", "pattern": "[ -~]{1,700}"},
                "prompts": {"type": "array", "items": prompt, "minItems": len(labels), "maxItems": len(labels)}})


def review_schema(labels, shapes, holes, sam_pending=False):
    ids = [shape["id"] for shape in shapes]
    hole_ids = [item["id"] for item in holes]
    identifier = {"type": "integer", "enum": ids}
    one = {"type": "array", "items": identifier, "minItems": 1, "maxItems": 1}
    polygon = {"type": "array", "minItems": 3, "maxItems": 32,
               "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}
    def operation(names, extra=None, shape_ids=one):
        return obj({"operation": {"type": "string", "enum": names}, "shape_ids": shape_ids,
                    **(extra or {}), "reason": reason()})
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
                                 "semantic": {"type": "string", "enum": ["ground", "non_ground"]}, "reason": reason()})}
    operations = {"type": "array", "maxItems": 3 if tools else 0,
                  "items": {"anyOf": tools} if tools else {"type": "object"}}
    properties = {"status": {"type": "string", "enum": ["continue", "done"]}, "reason": reason(),
                  "hole_assessment": assessments, "operations": operations}
    if sam_pending:
        properties["sam_assessment"] = obj({"improved": {"type": "boolean"}, "reason": reason()})
    return obj(properties)
