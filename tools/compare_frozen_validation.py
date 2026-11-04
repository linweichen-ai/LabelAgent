"""Check frozen adoption gates from complete V2 scores and unchanged candidates.

This consumes already scored reference agreement. It never creates predictions,
changes the metric, selects a different primary, or confirms annotation truth.
"""
from fractions import Fraction
import json


BASELINE='pallet_baseline'
PREVIOUS='four_crop_control'
RADII=('0','3','6')
COUNT_FIELDS=('reference_projected_points','supported_reference_objects','candidates','raw_mask_union_pixels')
RADIUS_COUNTS=('covered_projected_points','points_covered_by_other_class','objects_matched_at_support_threshold')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def count(value):
    if type(value) is not int or value<0:raise ValueError('Nonnegative integer score counts required')
    return value


def compact(summary):
    p,h=[summary['groups']['all'][label] for label in ('pallet','person')]
    return dict(pallet_objects_matched=p['radii']['3']['objects_matched_at_support_threshold'],
        pallet_reference_objects=p['supported_reference_objects'],pallet_points_covered=p['radii']['3']['covered_projected_points'],
        pallet_reference_points=p['reference_projected_points'],pallet_candidates=p['candidates'],
        pallet_union_pixels=p['raw_mask_union_pixels'],person_points_covered_by_pallet=h['radii']['3']['points_covered_by_other_class'],
        person_reference_points=h['reference_projected_points'])


def build_comparison(frozen,reports,per_image,candidate_maps,historical_summary,historical_rows):
    frames=frozen['full_validation_frames'] if 'full_validation_frames' in frozen else frozen['frames']
    cameras=frozen['cameras'];primary=frozen['primary_method'];policy=frozen['comparison_requires']
    for values in (frames,cameras):
        if not values or any(type(v) is not int or v<0 for v in values) or len(values)!=len(set(values)):
            raise ValueError('Frozen dimensions must contain distinct nonnegative integers')
    expected={(frame,camera) for frame in frames for camera in cameras};images=len(expected)
    if frozen.get('full_validation_images',images)!=images:raise ValueError('Frozen image count differs')
    required={BASELINE,PREVIOUS,primary}
    if not required<=reports.keys() or set(reports)!=set(per_image) or set(reports)!=set(candidate_maps):
        raise ValueError('Complete baseline, previous control and frozen primary required')
    if frozen.get('secondary_can_replace_primary',False):raise ValueError('Primary replacement is forbidden')
    if any(policy[key] is not True for key in ('strict_instance_gain_over_previous_best','point_coverage_not_below_previous_best',
        'person_off_class_not_above_previous_best','all_original_candidates_preserved')):
        raise ValueError('Original adoption requirements cannot be relaxed')
    numeric={'original_minimum_absolute_gain':'0.02','original_maximum_candidate_count_ratio':'1.5',
        'original_maximum_union_area_ratio':'1.25','original_person_off_class_increase_limit':'0.005'}
    if any(Fraction(str(policy[key]))!=Fraction(value) for key,value in numeric.items()):
        raise ValueError('Original numeric adoption thresholds cannot be changed')
    baseline_summary=reports[BASELINE]
    indexed={}
    def check_report(name,summary,rows):
        if (summary.get('status')!='complete' or summary.get('metric_version')!=2 or
            summary.get('planned_images')!=images or summary.get('evaluated_images')!=images or
            summary.get('missing_prediction_images')!=0):raise ValueError('Incomplete V2 score report')
        if (summary.get('archive_sha256')!=baseline_summary.get('archive_sha256') or
            summary.get('parameters')!=baseline_summary.get('parameters')):raise ValueError('Mixed datasets or metric parameters')
        if summary.get('reference_source')!='labels' or summary.get('reference_is_independent_ground_truth') is not False:
            raise ValueError('Reference scope differs')
        values={}
        for row in rows:
            if type(row['frame']) is not int or type(row['camera']) is not int:raise ValueError('Integer frame and camera identifiers required')
            key=(row['frame'],row['camera'])
            if key in values:raise ValueError('Duplicate scored image')
            values[key]=row['classes']
        if set(values)!=expected:raise ValueError('Missing or unexpected scored images')
        for label in ('pallet','person'):
            total=summary['groups']['all'][label]
            if total.get('images')!=images:raise ValueError('Incomplete class image coverage')
            for field in COUNT_FIELDS:
                if sum(count(v[label][field]) for v in values.values())!=count(total[field]):
                    raise ValueError('Summary counts differ from per-image readback')
            for radius in RADII:
                for field in RADIUS_COUNTS:
                    if sum(count(v[label]['radii'][radius][field]) for v in values.values())!=count(total['radii'][radius][field]):
                        raise ValueError('Summary radius counts differ from per-image readback')
        indexed[name]=values
    for name,summary in reports.items():check_report(name,summary,per_image[name])
    check_report('__historical_previous__',historical_summary,historical_rows)
    if (indexed[PREVIOUS]!=indexed['__historical_previous__'] or
        canonical(reports[PREVIOUS]['groups'])!=canonical(historical_summary['groups'])):
        raise ValueError('Previous four-crop control differs from historical scores')
    for values in indexed.values():
        for key,classes in values.items():
            for label in ('pallet','person'):
                for field in ('reference_projected_points','supported_reference_objects'):
                    if classes[label][field]!=indexed[BASELINE][key][label][field]:raise ValueError('Reference denominators differ')
    preserved={};point_readback={}
    for name,mapping in candidate_maps.items():
        prefix=True
        actual={(topic,frame) for topic,rows in mapping.items() for frame in rows}
        if name!=BASELINE and actual!={(f'/capture/camera_{c}/image',str(f)) for f,c in expected}:
            raise ValueError('Candidate export does not cover the frozen image cohort exactly')
        class_counts={'pallet':0,'person':0}
        for frame,camera in expected:
            topic=f'/capture/camera_{camera}/image';key=str(frame)
            original=candidate_maps[BASELINE][topic][key];values=mapping[topic][key]
            prefix=prefix and canonical(values[:len(original)])==canonical(original)
            for shape in values:
                if shape['label'] in class_counts:class_counts[shape['label']]+=1
        for label,value in class_counts.items():
            if value!=reports[name]['groups']['all'][label]['candidates']:raise ValueError('Candidate export counts differ from metric')
        preserved[name]=prefix
        point_readback[name]=all(v['pallet']['radii']['3']['covered_projected_points']>=
            indexed[PREVIOUS][key]['pallet']['radii']['3']['covered_projected_points'] for key,v in indexed[name].items())
    base=compact(baseline_summary);old=compact(reports[PREVIOUS])
    if any(base[k]<=0 for k in ('pallet_reference_objects','pallet_reference_points','person_reference_points')):
        raise ValueError('Nonempty reference denominators required')
    rows=[]
    for name,report in reports.items():
        value=compact(report)
        checks={'strict_instance_gain_over_previous_best':value['pallet_objects_matched']>old['pallet_objects_matched'],
            'point_coverage_not_below_previous_best':value['pallet_points_covered']>=old['pallet_points_covered'],
            'person_off_class_not_above_previous_best':value['person_points_covered_by_pallet']<=old['person_points_covered_by_pallet'],
            'original_minimum_two_percentage_point_gain':Fraction(value['pallet_objects_matched'],value['pallet_reference_objects'])>=
                Fraction(base['pallet_objects_matched'],base['pallet_reference_objects'])+Fraction('0.02'),
            'original_count_ratio_limit':value['pallet_candidates']<=base['pallet_candidates']*Fraction('1.5'),
            'original_union_area_limit':value['pallet_union_pixels']<=base['pallet_union_pixels']*Fraction('1.25'),
            'original_person_cross_class_limit':Fraction(value['person_points_covered_by_pallet'],value['person_reference_points'])<=
                Fraction(base['person_points_covered_by_pallet'],base['person_reference_points'])+Fraction('0.005'),
            'all_original_candidates_preserved':preserved[name]}
        passed=all(checks.values())
        rows.append(dict(method=name,**value,checks=checks,passed_numeric_and_preservation_gates=passed,
            raw_point_coverage_nonregression_every_image=point_readback[name],
            eligible=name==primary and passed and point_readback[name]))
    chosen=next(row for row in rows if row['method']==primary)
    return {'status':'complete','rows':rows,'primary_method':primary,'primary_passed':chosen['eligible'],
        'secondary_can_replace_primary':False,'winner':primary if chosen['eligible'] else None,
        'baseline':base,'previous_best':old,'scored_images_per_method':images,
        'historical_four_crop_all_classes_radii_and_groups_exact':True,
        'summary_counts_match_per_image_readback':True,'source_candidate_counts_match_scores':True,
        'adopted':False,'independent_external_ground_truth':False,
        'scope':'Frozen same-capture sparse reference agreement; not independent pixel accuracy'}
