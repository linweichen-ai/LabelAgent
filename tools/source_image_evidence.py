"""Conservative evidence availability for an unmarked annotation source crop.

This is an isolated annotation helper, not an object detector or VLM API filter.
The caller must supply the original unmarked crop, before outlines, text or cards.
"""
from dataclasses import asdict, dataclass
import hashlib
import io
import warnings

from PIL import Image, UnidentifiedImageError


POLICY = "unmarked-source-exact-uniform-v1"
CATEGORIES = frozenset(("pallet", "person", "uncertain"))


@dataclass(frozen=True)
class SourceEvidence:
    source_sha256: str
    available: bool
    reason: str
    size: tuple = ()
    source_mode: str = ""
    channel_ranges: tuple = ()


def inspect_source(source_bytes):
    """No classes, IDs, scores or expected answers are accepted as input.

    Nonuniform only means available for later review, never a valid object.
    Alpha, palette and multiframe inputs require explicit upstream handling.
    """
    if not isinstance(source_bytes, bytes):
        raise TypeError("Encoded unmarked source bytes required")
    digest = hashlib.sha256(source_bytes).hexdigest()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(source_bytes)) as image:
                size, mode = image.size, image.mode
                if getattr(image, "n_frames", 1) != 1:
                    return SourceEvidence(digest, False, "unsupported_multiframe", size, mode)
                if mode not in ("RGB", "L"):
                    return SourceEvidence(digest, False, "unsupported_mode", size, mode)
                image.load()
                ranges = tuple(high - low for low, high in image.convert("RGB").getextrema())
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombWarning, Image.DecompressionBombError):
        return SourceEvidence(digest, False, "unreadable_source")
    uniform = all(value == 0 for value in ranges)
    return SourceEvidence(digest, not uniform,
                          "exactly_uniform_source" if uniform else "nonuniform_source",
                          size, mode, ranges)


def guard_category(proposed, evidence):
    """Abstain on unavailable input; never generate an alternative class."""
    if proposed not in CATEGORIES:
        raise ValueError("Unsupported category")
    if not isinstance(evidence, SourceEvidence):
        raise TypeError("SourceEvidence required")
    category = proposed if evidence.available else "uncertain"
    return {"policy": POLICY, "raw_category": proposed, "category": category,
            "changed": category != proposed, "source_evidence": asdict(evidence)}
