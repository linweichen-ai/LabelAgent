"""Conservative subject veto; mentioning a pallet is not sufficient identity."""
import re
from semantic_evidence_gate import pallet_evidence
def compatible_pallet_subject(caption):
 text=caption.get('object','')
 return (pallet_evidence(caption) and not re.search(r'\bpallet\s+jack\b|\bforklift\b|\bpossibly\b|\bmaybe\b|\buncertain\b',text,re.I))
