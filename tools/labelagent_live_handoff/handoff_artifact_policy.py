"""Explicit local experiment roots; copy evidence without widening to a drive."""
import re
from pathlib import Path


def approved_artifact(path,workspace,extra_roots=()):
    try:
        resolved=Path(path).resolve();workspace=Path(workspace).resolve();relative=None
        try:
            rel=resolved.relative_to(workspace)
            if len(rel.parts)>=3 and rel.parts[0]=='outputs' and rel.parts[1].startswith('labelagent_'):relative=rel
        except ValueError:pass
        if relative is None:
            for root in extra_roots:
                try:
                    rel=resolved.relative_to(Path(root).resolve())
                    if len(rel.parts)>=1:relative=rel;break
                except ValueError:pass
        if relative is None:return None
        if re.search(r'cnb|auth|credential|token|login|browser.profile|browser_tmp',str(relative),re.I):return None
        if resolved.suffix.lower() in ['.json','.md','.txt'] and re.search(r'private',str(relative),re.I):return None
        return resolved if resolved.is_file() else None
    except (ValueError,OSError):return None
