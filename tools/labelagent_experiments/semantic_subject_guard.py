"""Conservative subject veto; mentioning a pallet is not sufficient identity."""
import re
from semantic_evidence_gate import pallet_evidence

def pallet_is_primary_subject(text):
 """Conservative textual subject check; not a visual identity proof."""
 if not isinstance(text,str):return False
 text=text.strip().lower()
 text=re.sub(r'^(?:in|at) the (?:center|foreground|background)(?: of (?:the|this) (?:image|photo))?\s*[,;:]?\s*','',text)
 text=re.sub(r'^(?:there (?:is|are)|(?:the|this) (?:central |centered |main |visible )?(?:object|surface) (?:is|appears to be))\s+','',text)
 primary=re.split(r'\b(?:with|and|on|in|under|over|above|below|beside|behind|next to|near|against|containing|carrying)\b|[,;.]',text,maxsplit=1)[0]
 return bool(re.search(r'\bpallets?\b|\bskids?\b|托盘',primary,re.I))

def compatible_pallet_subject(caption):
 text=caption.get('object','')
 return (pallet_evidence(caption) and pallet_is_primary_subject(text)
         and not re.search(r'\bpallet\s+jack\b|\bforklift\b|\bpossibly\b|\bmaybe\b|\buncertain\b',text,re.I))
