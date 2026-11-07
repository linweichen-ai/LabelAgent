"""Measure SAM mask coverage of visible, class-labelled LiDAR projections.

Uses the selected 3D labels and supplied camera calibration. This is a sparse
reference-agreement measure, never dense pixel accuracy. Unknown/ambiguous
points are not treated as negative segmentation truth.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import yaml

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from compare_boxes3d import Box, match
from import_capture import read_pcd
from run_local_agent_acceptance import load_image, save
from source.agent.edit_agent import _rasterize, valid_shape

LABEL_BITS={'Pallet':1,'Pedestrian':2}
LABEL_NAMES={1:'pallet',2:'person'}
TUNING_FRAMES={15,45,76,105,135}


def checked_source(dataset, manifest, relative):
    root=(dataset/'source').resolve()
    path=(root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source path escapes dataset')
    raw=path.read_bytes()
    expected=manifest['extracted_sha256'].get(relative)
    if expected is None or hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Source differs from import manifest')
    return raw


def camera_parameters(intrinsic, extrinsic):
    if intrinsic.get('model_type')!='PINHOLE' or extrinsic.get('calibrated') is not True:
        raise ValueError('Requires an explicitly calibrated PINHOLE camera')
    distortions=np.array([float(value) for value in intrinsic.get('distortion_parameters',{}).values()])
    if not np.isfinite(distortions).all() or np.any(distortions!=0):
        raise ValueError('Nonzero distortion is not silently ignored')
    width,height=int(intrinsic['image_width']),int(intrinsic['image_height'])
    params=intrinsic['projection_parameters']
    vector=np.array([params[key] for key in ('fx','fy','cx','cy')],dtype=float)
    transform=np.array(extrinsic['transform:'],dtype=float).reshape(4,4)
    if width<=0 or height<=0 or not np.isfinite(vector).all() or (vector[:2]<=0).any() or not np.isfinite(transform).all():
        raise ValueError('Invalid camera parameters')
    rotation=transform[:3,:3]
    if (not np.allclose(transform[3],[0,0,0,1],atol=1e-8) or
            abs(np.linalg.det(rotation)-1)>.01 or np.linalg.norm(rotation.T@rotation-np.eye(3))>.01):
        raise ValueError('Extrinsic is not a rigid transform')
    return {'width':width,'height':height,'intrinsics':vector,'transform':transform}


def class_membership(points, objects):
    """Do not choose an instance for points inside overlapping reference boxes."""
    bits=np.zeros(len(points),dtype=np.uint8)
    owner=np.zeros(len(points),dtype=np.int32)
    boxes=[]
    for obj in objects:
        if obj.get('ignore') or obj.get('obj_type') not in LABEL_BITS:
            continue
        box=Box(obj)
        inside=(np.abs((points-box.center)@box.rotation)<=box.size/2+1e-8).all(axis=1)
        identifier=len(boxes)+1
        owner[inside]=np.where(owner[inside]==0,identifier,-1)
        bits[inside]|=LABEL_BITS[box.label]
        boxes.append({'box':box,'obj_id':obj.get('obj_id'),'bit':LABEL_BITS[box.label]})
    return bits,owner,boxes


def project_visible(points, camera, cell_size=8, depth_slack=.15, near=.2):
    if cell_size<1 or depth_slack<0 or near<=0:
        raise ValueError('Invalid visibility parameters')
    transform=camera['transform']
    xyz=points@transform[:3,:3].T+transform[:3,3]
    good=np.isfinite(xyz).all(axis=1)&(xyz[:,2]>near)
    source=np.flatnonzero(good)
    xyz=xyz[good]
    fx,fy,cx,cy=camera['intrinsics']
    uv=xyz[:,:2]/xyz[:,2,None]*[fx,fy]+[cx,cy]
    inside=(uv[:,0]>=0)&(uv[:,0]<camera['width']-.5)&(uv[:,1]>=0)&(uv[:,1]<camera['height']-.5)
    source,xyz,uv=source[inside],xyz[inside],uv[inside]
    pixels=np.floor(uv+.5).astype(np.int32)
    columns=(camera['width']+cell_size-1)//cell_size
    rows=(camera['height']+cell_size-1)//cell_size
    cells=(pixels[:,1]//cell_size)*columns+pixels[:,0]//cell_size
    nearest=np.full(rows*columns,np.inf)
    np.minimum.at(nearest,cells,xyz[:,2])
    visible=xyz[:,2]<=nearest[cells]+depth_slack
    return source[visible],pixels[visible]


def mask_support(shapes, source_indices, pixels, bits, owner, boxes, width, height, radii=(0,3,6), minimum_points=5, minimum_purity=.67):
    # Archived SAM3 exports use string IDs; this metric addresses rows, not
    # editable workbench IDs. Validate a copy with an integer ID only.
    if any(shape.get('label') not in LABEL_NAMES.values() or not valid_shape({**shape,'id':index+1},width,height)
           for index,shape in enumerate(shapes)):
        raise ValueError('Prediction has an unsupported label or invalid geometry')
    if len(shapes)>400:
        raise ValueError('Unexpectedly many candidates')
    result={'classes':{},'objects':[],
            'visible_projected_points':len(source_indices),
            'cross_class_ambiguous_points':int((bits[source_indices]==3).sum())}
    owners_visible=owner[source_indices]
    object_indices={i:np.flatnonzero(owners_visible==i+1) for i in range(len(boxes))}
    eligible={i:indices for i,indices in object_indices.items() if len(indices)>=minimum_points}
    semantic=bits[source_indices]
    other_unions={}
    for bit,label in LABEL_NAMES.items():
        masks=[_rasterize(shape,width,height) for shape in shapes if shape['label']==label]
        union=np.zeros((height,width),dtype=np.uint8)
        for mask in masks:union|=mask
        other_unions[label]=union
    for bit,label in LABEL_NAMES.items():
        mask_list=[_rasterize(shape,width,height) for shape in shapes if shape['label']==label]
        point_rows=np.flatnonzero(semantic==bit)
        unique_class_rows=np.flatnonzero((semantic==bit)&(owners_visible>0))
        indices=[i for i in eligible if boxes[i]['bit']==bit]
        statistics={'reference_projected_points':len(point_rows),'supported_reference_objects':len(indices),
                    'candidates':len(mask_list),'raw_mask_union_pixels':int(other_unions[label].sum()),'radii':{}}
        for radius in radii:
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
            union=np.zeros((height,width),dtype=np.uint8)
            matrix=np.zeros((len(indices),len(mask_list)))
            raw_coverage=np.zeros_like(matrix)
            for column,mask in enumerate(mask_list):
                dilated=cv2.dilate(mask,kernel) if radius else mask
                union|=dilated
                known_samples=pixels[unique_class_rows]
                known_hits=int(dilated[known_samples[:,1],known_samples[:,0]].sum()) if len(known_samples) else 0
                for row,identifier in enumerate(indices):
                    samples=pixels[eligible[identifier]]
                    hits=int(dilated[samples[:,1],samples[:,0]].sum())
                    coverage=hits/len(samples)
                    purity=hits/known_hits if known_hits else 0
                    raw_coverage[row,column]=coverage
                    matrix[row,column]=coverage if purity>=minimum_purity else 0
            opposite='person' if label=='pallet' else 'pallet'
            other=cv2.dilate(other_unions[opposite],kernel) if radius else other_unions[opposite]
            samples=pixels[point_rows]
            covered=int(union[samples[:,1],samples[:,0]].sum()) if len(samples) else 0
            off_class=int(other[samples[:,1],samples[:,0]].sum()) if len(samples) else 0
            pairs=match(matrix,.5)
            best=matrix.max(axis=1) if len(mask_list) else np.zeros(len(indices))
            statistics['radii'][str(radius)]={'covered_projected_points':covered,
                'points_covered_by_other_class':off_class,'objects_matched_at_support_threshold':len(pairs),
                'best_candidate_support_sum':float(best.sum())}
            if radius==3:
                for row,identifier in enumerate(indices):
                    result['objects'].append({'obj_id':boxes[identifier]['obj_id'],'label':label,
                        'unique_visible_points':len(eligible[identifier]),'best_candidate_supported_coverage':float(best[row]),
                        'best_candidate_point_coverage':float(raw_coverage[row].max()) if len(mask_list) else 0})
        result['classes'][label]=statistics
    return result


def verify_prediction_receipt(raw, receipt, manifest):
    if (hashlib.sha256(raw).hexdigest()!=receipt.get('prediction_json_sha256') or
            receipt.get('archive_sha256')!=manifest['archive_sha256']):
        raise ValueError('Prediction receipt does not bind these candidates and dataset')


def evaluate(args):
    dataset=Path(args.dataset).resolve()
    manifest=json.loads((dataset/'import_manifest.json').read_text(encoding='utf-8'))
    raw_predictions=Path(args.candidates_json).read_bytes()
    receipt=json.loads(Path(args.prediction_receipt).read_text(encoding='utf-8'))
    verify_prediction_receipt(raw_predictions,receipt,manifest)
    predictions=json.loads(raw_predictions)
    frames=(list(range(len(manifest['timestamps_ns']))) if args.frames=='all' else [int(value) for value in args.frames.split(',')])
    cameras=[int(value) for value in args.cameras.split(',')]
    if not frames or len(set(frames))!=len(frames) or any(frame<0 or frame>=len(manifest['timestamps_ns']) for frame in frames):
        raise ValueError('Invalid frame selection')
    if not cameras or len(set(cameras))!=len(cameras) or any(camera not in range(4) for camera in cameras):
        raise ValueError('Invalid camera selection')
    radii=[int(value) for value in args.radii.split(',')]
    if not radii or len(set(radii))!=len(radii) or any(radius<0 or radius>16 for radius in radii):
        raise ValueError('Invalid pixel tolerance')
    if (args.cell_size<1 or args.cell_size>32 or not 0<=args.depth_slack<=1 or args.minimum_object_points<1
            or not 0<=args.minimum_object_purity<=1):
        raise ValueError('Invalid reference-support parameters')
    extrinsics=yaml.safe_load(checked_source(dataset,manifest,'calib/lidar2camera.yaml'))
    camera_info={}
    for camera in cameras:
        text=checked_source(dataset,manifest,f'calib/camera{camera}_calib.yaml').decode('utf-8')
        intrinsic=yaml.safe_load('\n'.join(line for line in text.splitlines() if not line.startswith('%YAML:')))
        camera_info[camera]=camera_parameters(intrinsic,extrinsics[f'lidar_to_camera_{camera}'])
    records={(row['frame'],row['topic']):row for row in manifest['records'] if row['topic'].endswith('/image')}
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'sparse_projected_reference_agreement_not_pixel_accuracy','metric_version':2,'archive_sha256':manifest['archive_sha256'],
        'prediction_json_sha256':receipt['prediction_json_sha256'],'planned_images':len(frames)*len(cameras),
        'evaluated_images':0,'missing_prediction_images':0,'pixel_accuracy_evaluated':False,
        'reference_source':'labels','reference_is_independent_ground_truth':False,
        'parameters':{'radii_pixels':radii,'zbuffer_cell_pixels':args.cell_size,'depth_slack_m':args.depth_slack,
                      'near_plane_m':.2,'minimum_unique_object_points':args.minimum_object_points,
                      'reference_point_inside_box_tolerance_m':0,'one_to_one_object_support_threshold':.5,
                      'minimum_object_point_purity':args.minimum_object_purity,
                      'tuning_frames':sorted(TUNING_FRAMES)},
        'coordinate_convention':'labels in lidar frame; provided lidar-to-camera PINHOLE transforms; no trajectory pose applied',
        'rotation_convention':'Rz(yaw) Ry(pitch) Rx(roll), radians, inferred from field names',
        'unlabelled_or_cross_class_ambiguous_points_are_negatives':False,'groups':{}}
    save(output/'summary.json',report)
    detailed=[]
    accum=defaultdict(lambda:defaultdict(lambda:{'images':0,'reference_projected_points':0,'supported_reference_objects':0,
        'candidates':0,'raw_mask_union_pixels':0,'radii':{str(radius):{'covered_projected_points':0,
            'points_covered_by_other_class':0,'objects_matched_at_support_threshold':0,'best_candidate_support_sum':0.0,
            'image_point_coverage_sum':0.0,'images_with_reference_points':0} for radius in radii}}))
    try:
        for frame in frames:
            stamp=str(manifest['timestamps_ns'][frame])
            raw_label=checked_source(dataset,manifest,f'labels/{stamp}.json')
            annotation=json.loads(raw_label)
            if str(annotation.get('frame_id'))!=stamp:
                raise ValueError('Annotation timestamp mismatch')
            relative=f'lidar/{stamp}.pcd'
            checked_source(dataset,manifest,relative)
            payload,fields,stride,count,_=read_pcd(dataset/'source'/relative)
            fields={field['name']:field for field in fields}
            types={7:'<f4',8:'<f8'}
            xyz=np.column_stack([np.ndarray((count,),dtype=types[fields[name]['type']],buffer=payload,
                offset=fields[name]['offset'],strides=(stride,)) for name in ['x','y','z']])
            xyz=xyz[np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)>.001)]
            bits,owner,boxes=class_membership(xyz,annotation['objects'])
            for camera in cameras:
                topic=f'/capture/camera_{camera}/image'
                record=records[frame,topic]
                if str(frame) not in predictions.get(topic,{}):
                    report['missing_prediction_images']+=1
                    continue
                if receipt.get('image_sha256',{}).get(topic,{}).get(str(frame))!=record['sha256']:
                    raise ValueError('Predictions are not bound to the requested source image')
                raw_image=load_image(dataset,record)
                size=Image.open(io.BytesIO(raw_image)).size
                info=camera_info[camera]
                if size!=(info['width'],info['height']):
                    raise ValueError('Image dimensions differ from supplied calibration')
                source,pixels=project_visible(xyz,info,args.cell_size,args.depth_slack)
                value=mask_support(predictions[topic][str(frame)],source,pixels,bits,owner,boxes,*size,radii,args.minimum_object_points,args.minimum_object_purity)
                detailed.append({'frame':frame,'camera':camera,**value})
                report['evaluated_images']+=1
                for group in ('all','tuning' if frame in TUNING_FRAMES else 'validation',f'camera_{camera}'):
                    for label,stats in value['classes'].items():
                        target=accum[group][label]
                        target['images']+=1
                        for key in ('reference_projected_points','supported_reference_objects','candidates','raw_mask_union_pixels'):
                            target[key]+=stats[key]
                        for radius,row in stats['radii'].items():
                            for key in ('covered_projected_points','points_covered_by_other_class','objects_matched_at_support_threshold','best_candidate_support_sum'):
                                target['radii'][radius][key]+=row[key]
                            if stats['reference_projected_points']:
                                target['radii'][radius]['image_point_coverage_sum']+=row['covered_projected_points']/stats['reference_projected_points']
                                target['radii'][radius]['images_with_reference_points']+=1
            if frame%8==0:
                save(output/'summary.json',report)
                print(json.dumps({'frame':frame,'evaluated_images':report['evaluated_images']}),flush=True)
        for group,labels in accum.items():
            report['groups'][group]={}
            for label,stats in labels.items():
                for row in stats['radii'].values():
                    points,objects=stats['reference_projected_points'],stats['supported_reference_objects']
                    row['projected_point_coverage']=row['covered_projected_points']/points if points else None
                    row['off_class_coverage']=row['points_covered_by_other_class']/points if points else None
                    row['one_to_one_object_support_fraction']=row['objects_matched_at_support_threshold']/objects if objects else None
                    row['mean_best_object_support']=row['best_candidate_support_sum']/objects if objects else None
                    row['mean_image_point_coverage']=row['image_point_coverage_sum']/row['images_with_reference_points'] if row['images_with_reference_points'] else None
                report['groups'][group][label]=stats
        report['status']='complete'
        report['input_checksums_verified']=True
        save(output/'per_image.json',detailed)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error),encoding='utf-8')
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(output/'summary.json',report)
    print(json.dumps({'status':report['status'],'evaluated_images':report['evaluated_images'],
        'all':report['groups'].get('all')},indent=2))
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--candidates-json',required=True)
    parser.add_argument('--prediction-receipt',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--frames',default='all')
    parser.add_argument('--cameras',default='0,1,2,3')
    parser.add_argument('--radii',default='0,3,6')
    parser.add_argument('--cell-size',type=int,default=8)
    parser.add_argument('--depth-slack',type=float,default=.15)
    parser.add_argument('--minimum-object-points',type=int,default=5)
    parser.add_argument('--minimum-object-purity',type=float,default=.67)
    raise SystemExit(evaluate(parser.parse_args()))
