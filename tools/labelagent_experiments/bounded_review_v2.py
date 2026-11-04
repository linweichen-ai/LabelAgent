"""Assess every supplied hole in bounded real-review batches; cap edits per round."""
from copy import deepcopy
import io
import json


def install_review_batches(workflow, batch_size=10, operation_cap=3, callback=None):
    original = workflow.review

    def batched(config, image, overlay, description, shapes, labels, history, **kwargs):
        holes = kwargs.get('holes') or []
        if len(holes) <= batch_size:
            return original(config, image, overlay, description, shapes, labels, history, **kwargs)
        decisions = []
        for index in range(0, len(holes), batch_size):
            # Overview and crop evidence keep the full-round numbering.
            part = [dict(hole, number=index+offset+1)
                    for offset,hole in enumerate(holes[index:index + batch_size])]
            local = dict(kwargs, holes=part)
            prompt = deepcopy(description)
            prompt['description'] += (' This request assesses ONLY the listed hole IDs. Other numbered holes remain '
                                      'visible for context and are assessed in separate batches. Never invent missing IDs.')
            decision = original(config, image, overlay, prompt, shapes, labels, history, **local)
            decisions.append(decision)
            if callback:
                callback({'batch':len(decisions),'round':len(history)+1,
                          'hole_ids':[h['id'] for h in part],'decision':deepcopy(decision)})
        assessment = [a for decision in decisions for a in decision.get('hole_assessment', [])]
        expected = {h['id'] for h in holes}
        if len(assessment) != len(holes) or {a.get('hole_id') for a in assessment} != expected:
            raise ValueError('Batched review did not cover every original hole exactly once')
        operations, seen = [], set()
        for decision in decisions:
            for operation in decision.get('operations', []):
                key = json.dumps(operation, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    operations.append(operation); seen.add(key)
        selected = operations[:operation_cap]
        merged = {'status':'continue' if selected or any(d['status']=='continue' for d in decisions) else 'done',
                  'reason':'Batched real model review; '+ '; '.join(d.get('reason','') for d in decisions)[:500],
                  'hole_assessment':assessment,'operations':selected}
        comparison = kwargs.get('sam_comparison')
        if comparison is not None:
            improved = all(d.get('sam_assessment', {}).get('improved') is True for d in decisions)
            merged['sam_assessment'] = {'improved':improved,
                'reason':'Every batch confirms improvement' if improved else 'At least one batch does not confirm improvement'}
        from PIL import Image
        width, height = Image.open(io.BytesIO(image)).size
        merged = workflow._validate_review_result(merged, holes, shapes, labels, width, height, comparison)
        if callback:
            callback({'merged':True,'batches':len(decisions),'round':len(history)+1,
                      'all_holes_assessed':True,'original_holes':len(holes),
                      'proposed_operations':len(operations),'selected_operations':len(selected),
                      'deferred_operations':max(0,len(operations)-len(selected))})
        return merged

    workflow.review = batched
    return original
