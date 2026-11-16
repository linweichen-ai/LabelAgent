"""Index camera messages and decode H.264 or image payloads from MCAP."""
import io
import hashlib
import json
import math
from fractions import Fraction
import threading
from bisect import bisect_right
from pathlib import Path

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image
import av
from .sensors import catalog, message_time


class Video:
    CATALOG_CACHE_VERSION = 1

    def __init__(self, path, index_directory=None):
        self.path = Path(path)
        if not self.path.is_file() or self.path.suffix.lower() != ".mcap":
            raise ValueError("请选择存在的 .mcap 文件")
        self.topics = {}
        self.cache = {}
        self._decoder_state = {}
        self._keyframes = {}
        self._lock = threading.RLock()
        self._playback = {}
        self._playback_lock = threading.Lock()
        cached = self._read_catalog_cache(index_directory)
        if cached is None:
            self.catalog, self.origin_ns, self.duration = catalog(self.path)
            self._write_catalog_cache(index_directory)
        else:
            self.catalog, self.origin_ns, self.duration = cached
        self.topics = {topic: info for topic, info in self.catalog.items() if info['type'] == 'image'}

    def _catalog_signature(self):
        stat = self.path.stat()
        return {
            "version": self.CATALOG_CACHE_VERSION,
            "path": str(self.path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _catalog_cache_path(self, index_directory):
        if index_directory is None:
            return None
        key = hashlib.sha256(str(self.path.resolve()).encode()).hexdigest()
        return Path(index_directory) / f"{key}.json"

    def _read_catalog_cache(self, index_directory):
        target = self._catalog_cache_path(index_directory)
        if target is None or not target.is_file() or target.stat().st_size > 1024 * 1024:
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if payload.get("signature") != self._catalog_signature():
                return None
            cached_catalog = payload.get("catalog")
            origin_ns = payload.get("origin_ns")
            duration = payload.get("duration")
            if (not isinstance(cached_catalog, dict) or isinstance(origin_ns, bool)
                    or not isinstance(origin_ns, int) or isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration < 0):
                return None
            for topic, info in cached_catalog.items():
                if (not isinstance(topic, str) or not isinstance(info, dict)
                        or info.get("type") not in {"image", "pointcloud", "other"}
                        or isinstance(info.get("count"), bool)
                        or not isinstance(info.get("count"), int)
                        or info["count"] < 0):
                    return None
            return cached_catalog, origin_ns, float(duration)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _write_catalog_cache(self, index_directory):
        target = self._catalog_cache_path(index_directory)
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(f".{threading.get_ident()}.tmp")
            temporary.write_text(json.dumps({
                "signature": self._catalog_signature(),
                "catalog": self.catalog,
                "origin_ns": self.origin_ns,
                "duration": self.duration,
            }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            temporary.replace(target)
        except OSError:
            # The index is an optimization only; media loading must still work.
            try:
                temporary.unlink(missing_ok=True)
            except (OSError, UnboundLocalError):
                pass

    @staticmethod
    def _image(data, payload):
        if payload.startswith((b"\xff\xd8", b"\x89PNG")):
            return Image.open(io.BytesIO(payload)).convert("RGB")
        if hasattr(data, 'width') and hasattr(data, 'height'):
            encoding = getattr(data, 'encoding', '')
            raw_mode = {'rgb8': 'RGB', 'bgr8': 'BGR', 'rgba8': 'RGBA', 'bgra8': 'BGRA', 'mono8': 'L'}.get(encoding.lower())
            if not raw_mode:
                raise ValueError(f"暂不支持原始图像编码：{encoding}")
            mode = 'RGBA' if 'A' in raw_mode else 'L' if raw_mode == 'L' else 'RGB'
            stride = getattr(data, 'step', 0) or data.width * len(raw_mode)
            return Image.frombytes(mode, (data.width, data.height), payload, 'raw', raw_mode, stride).convert('RGB')
        return None

    def frame(self, topic, index):
        if topic not in self.topics or not 0 <= index < self.topics[topic]["count"]:
            raise ValueError("无效的相机话题或帧号")
        key = (topic, index)
        with self._lock:
            if key in self.cache:
                return self.cache[key]
            state = self._decoder_state.get(topic)
            if state is None or index < state["next_index"] or index - state["next_index"] > 32:
                if state is not None:
                    state["stream"].close()
                stream = self.path.open("rb")
                reader = make_reader(stream)
                start_index, start_time = (0, None)
                if index >= 32:
                    start_index, start_time = self._keyframe_before(topic, index)
                state = {"stream": stream, "reader": reader,
                         "messages": reader.iter_messages(topics=[topic], start_time=start_time),
                         "codec": av.CodecContext.create("h264", "r"),
                         "factory": DecoderFactory(), "next_index": start_index}
                self._decoder_state[topic] = state
            while state["next_index"] <= index:
                schema, channel, message = next(state["messages"])
                decoder = state["factory"].decoder_for(channel.message_encoding, schema)
                data = decoder(message.data) if decoder else message.data
                payload = getattr(data, "data", data if isinstance(data, bytes) else None)
                if payload is None:
                    continue
                image = self._image(data, payload)
                images = [image] if image is not None else state["codec"].decode(av.Packet(payload))
                for decoded in images:
                    current = state["next_index"]
                    state["next_index"] += 1
                    if current == index:
                        buffer = io.BytesIO()
                        image = decoded.to_image() if isinstance(decoded, av.VideoFrame) else decoded
                        image.save(buffer, "JPEG", quality=90)
                        result = buffer.getvalue()
                        self.cache[key] = result
                        if len(self.cache) > 16:
                            self.cache.pop(next(iter(self.cache)))
                        return result
        raise ValueError("无法解码该帧")

    def _keyframe_before(self, topic, index):
        if index < 32:
            return 0, None
        if topic not in self._keyframes:
            frames = [(0, None)]
            factory = DecoderFactory()
            with self.path.open("rb") as stream:
                for frame_index, (schema, channel, message) in enumerate(
                    make_reader(stream).iter_messages(topics=[topic])
                ):
                    decoder = factory.decoder_for(channel.message_encoding, schema)
                    data = decoder(message.data) if decoder else message.data
                    payload = getattr(data, "data", data if isinstance(data, bytes) else b"")
                    if payload and ((hasattr(data, "width") and hasattr(data, "height")) or payload.startswith((b"\xff\xd8", b"\x89PNG")) or
                                    b"\x00\x00\x00\x01\x65" in payload or
                                    b"\x00\x00\x01\x65" in payload):
                        if frame_index:
                            frames.append((frame_index, message.log_time))
            self._keyframes[topic] = frames
        frames = self._keyframes[topic]
        position = bisect_right([frame for frame, _ in frames], index) - 1
        return frames[position]

    def playback(self, topic, directory):
        """Prepare a timestamped browser video once, without blocking frame requests."""
        if topic not in self.topics:
            raise ValueError("无效的相机话题")
        with self._playback_lock:
            if topic not in self._playback:
                stat = self.path.stat()
                key = hashlib.sha256(f"v1:{self.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{topic}".encode()).hexdigest()
                directory = Path(directory)
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / (key + ".mp4")
                manifest = directory / (key + ".json")
                if target.exists() and manifest.exists():
                    self._playback[topic] = dict(json.loads(manifest.read_text()), status="ready", url=f"/api/media/{key}.mp4")
                else:
                    self._playback[topic] = {"status": "preparing", "progress": 0}
                    threading.Thread(target=self._prepare_playback, args=(topic, target, manifest), daemon=True).start()
            return dict(self._playback[topic])

    def _prepare_playback(self, topic, target, manifest):
        temporary = target.with_suffix(".partial.mp4")
        times = []
        factory = DecoderFactory()
        output = None
        source = None
        try:
            output = av.open(str(temporary), "w", options={"movflags": "+faststart"})
            video_stream = None
            base = None
            previous = -1
            fallback_count = 0
            time_base = Fraction(1, 1000000)
            with self.path.open("rb") as stream:
                for index, (schema, channel, message) in enumerate(make_reader(stream).iter_messages(topics=[topic])):
                    decoder = factory.decoder_for(channel.message_encoding, schema)
                    data = decoder(message.data) if decoder else message.data
                    payload = getattr(data, "data", data if isinstance(data, bytes) else None)
                    if not payload:
                        raise ValueError("相机消息缺少图像数据")
                    ns = message_time(data, message)
                    if not ns:
                        ns = message.log_time
                        fallback_count += 1
                    if base is None:
                        base = ns
                    pts = (ns - base) // 1000
                    if pts <= previous:
                        raise ValueError("相机时间戳未严格递增，无法准确播放")
                    image = self._image(data, payload)
                    is_image = image is not None
                    if video_stream is None:
                        if is_image:
                            video_stream = output.add_stream("libx264", rate=30)
                            video_stream.width, video_stream.height = image.size
                            video_stream.pix_fmt = "yuv420p"
                            video_stream.options = {"preset": "veryfast", "crf": "18", "bf": "0"}
                        else:
                            source = av.open(io.BytesIO(payload), format="h264")
                            video_stream = output.add_stream_from_template(source.streams.video[0])
                        video_stream.time_base = time_base
                        if is_image:
                            video_stream.codec_context.time_base = time_base
                    if is_image:
                        decoded = av.VideoFrame.from_image(image)
                        decoded.pts, decoded.time_base = pts, time_base
                        for packet in video_stream.encode(decoded):
                            output.mux(packet)
                    else:
                        packet = av.Packet(payload)
                        packet.stream = video_stream
                        packet.time_base = time_base
                        packet.pts = packet.dts = pts
                        packet.duration = pts - previous if previous >= 0 else 33333
                        packet.is_keyframe = b"\x00\x00\x01\x65" in payload
                        output.mux(packet)
                    times.append(pts / 1000000)
                    previous = pts
                    if index % 100 == 0:
                        self._playback[topic]["progress"] = index / self.topics[topic]["count"]
                if video_stream is None:
                    raise ValueError("相机话题为空")
                if is_image:
                    for packet in video_stream.encode():
                        output.mux(packet)
            output.close()
            output = None
            temporary.replace(target)
            metadata = {"times": times, "fps": (len(times)-1)/times[-1] if len(times)>1 and times[-1] else 0,
                        "duration": times[-1] if times else 0, "count": len(times),
                        "timestamp_source": "log_time" if fallback_count == len(times) else "capture", "progress": 1}
            manifest.write_text(json.dumps(metadata))
            self._playback[topic] = dict(metadata, status="ready", url=f"/api/media/{target.name}")
        except Exception as error:
            self._playback[topic] = {"status": "error", "error": str(error)}
        finally:
            if output is not None:
                output.close()
            if source is not None:
                source.close()
            temporary.unlink(missing_ok=True)
