"""Open ordinary photographs alongside the existing MCAP recordings."""
import hashlib
import io
import threading
from pathlib import Path

import av
from PIL import Image, ImageOps, UnidentifiedImageError

from .mcap_video import Video

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | {".mcap"}


def read_image(path):
    try:
        with Image.open(path) as original:
            if original.format not in {"JPEG", "PNG"}:
                raise ValueError("请选择有效的 JPG、JPEG 或 PNG 图片")
            return ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("图片无法读取，请选择完整的 JPG、JPEG 或 PNG 文件") from error


class StillImage:
    """A single full-resolution image with the same frame interface as a recording."""
    def __init__(self, path):
        self.path = Path(path)
        self.image = read_image(self.path)
        output = io.BytesIO()
        self.image.save(output, "PNG")
        self._data = output.getvalue()
        self.catalog = {"图片": dict(type="image", schema="LabelAgent.StillImage",
                                   encoding="png", count=1, offset=0, log_offset=0)}
        self.topics = self.catalog
        self.origin_ns = 0
        self.duration = 0
        self._playback = None
        self._lock = threading.Lock()

    def frame(self, topic, index):
        if topic not in self.topics or index != 0:
            raise ValueError("无效的图片或帧号")
        return self._data

    def playback(self, topic, directory):
        if topic not in self.topics:
            raise ValueError("无效的图片")
        with self._lock:
            return self._preview(directory)

    def _preview(self, directory):
        if self._playback is None:
            # Only the viewing preview is padded for H.264; annotation uses the
            # original-size PNG above, preserving odd sizes and pixel coordinates.
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(b"still-preview-v1:" + self._data).hexdigest()
            target = directory / (key + ".mp4")
            if not target.exists():
                width, height = self.image.size
                preview = Image.new("RGB", (width + width % 2, height + height % 2))
                preview.paste(self.image)
                temporary = target.with_suffix(".partial.mp4")
                try:
                    with av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart"}) as output:
                        stream = output.add_stream("libx264", rate=1)
                        stream.width, stream.height = preview.size
                        stream.pix_fmt = "yuv420p"
                        for packet in stream.encode(av.VideoFrame.from_image(preview)):
                            output.mux(packet)
                        for packet in stream.encode():
                            output.mux(packet)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            self._playback = dict(status="ready", times=[0], fps=0, duration=0,
                                  count=1, progress=1, url=f"/api/media/{key}.mp4")
        return dict(self._playback)


def open_media(path):
    path = Path(path)
    return StillImage(path) if path.suffix.lower() in IMAGE_SUFFIXES else Video(path)
