"""Unlabelled same-frame raw-point evidence from multiple predicted camera masks."""
import numpy as np

def ranking_inputs(baseline,anchors,proposals,mode):
    if mode=='baseline':return baseline,proposals
    if mode!='four':raise ValueError('Unknown ranking context')
    identifiers={s['id'] for s in anchors}
    return anchors,[s for s in proposals if s['id'] not in identifiers]

def merge_point_views(total_points,views):
    if type(total_points) is not int or total_points<0 or len(views)>4:raise ValueError('Finite four-camera point cohort required')
    visible=np.zeros(total_points,np.int16);person=np.zeros(total_points,np.int16);per_view={}
    for camera,view in views.items():
        indices=np.asarray(view['indices']);hits=np.asarray(view['person_hits'])
        if indices.ndim!=1 or indices.dtype.kind not in 'iu' or hits.dtype!=np.bool_ or hits.shape!=indices.shape:
            raise ValueError('Aligned integer raw indices and boolean hits required')
        if np.any(indices<0) or np.any(indices>=total_points) or len(np.unique(indices))!=len(indices):raise ValueError('Distinct in-range raw indices required')
        mask=np.zeros(total_points,bool);mask[indices[hits]]=True
        per_view[camera]=mask;visible[indices]+=1;person[indices[hits]]+=1
    return {'visible_counts':visible,'person_counts':person,'person_any':person>0,'per_view':per_view}

def rejects_novel_points(candidate_hits,anchor_hits,raw_indices,person_any):
    candidate=np.asarray(candidate_hits);anchor=np.asarray(anchor_hits);indices=np.asarray(raw_indices);people=np.asarray(person_any)
    if (candidate.dtype!=np.bool_ or anchor.dtype!=np.bool_ or candidate.ndim!=1 or candidate.shape!=anchor.shape
        or indices.dtype.kind not in 'iu' or indices.shape!=candidate.shape or people.dtype!=np.bool_ or people.ndim!=1
        or np.any(indices<0) or np.any(indices>=len(people))):raise ValueError('Aligned raw-point evidence required')
    return bool(np.any(people[indices[candidate&~anchor]]))
