"""Conservatively veto repeated-base evidence even if status says single."""
import re
from instance_evidence_views import INSTANCE_PROMPT,INSTANCE_SCHEMA,png,target_views
REPEATED=re.compile(r'\b(stack(?:ed|s)?|multiple|several|repeated|pallets|bases)\b',re.I)
def repeated_evidence(d):
 return bool(REPEATED.search(' '.join(str(d.get(k,'')) for k in ['object','evidence'])))
def single_physical_base(d):
 from semantic_subject_guard import compatible_pallet_subject
 # A schema enum may be echoed as an object noun. Normalize its separator;
 # the unchanged neutral-photo gate remains independently mandatory.
 subject=str(d.get('object','')).replace('_',' ')
 evidence=str(d.get('evidence','')).strip()
 return (d.get('category')=='pallet_base' and d.get('instance_status')=='single'
         and compatible_pallet_subject({'object':subject}) and bool(evidence)
         and not repeated_evidence(d))
def repeated_control_rejected(d):
 return d.get('category')=='pallet_base' and (d.get('instance_status')=='repeated' or repeated_evidence(d)) and not single_physical_base(d)
