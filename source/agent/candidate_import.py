"""Validate and merge external 2-D candidates without replacing saved work."""
import copy
import io
from PIL import Image
from .edit_agent import valid_shape


def prepare_candidate_import(video, by_topic, labels):
    if not isinstance(by_topic, dict) or not by_topic:
        raise ValueError("候选必须按图像话题和帧号组织")
    prepared = {}
    for topic, frames in by_topic.items():
        if topic not in video.topics or not isinstance(frames, dict):
            raise ValueError("候选包含无效图像话题")
        # The existing video playback contract uses a fixed resolution per topic.
        with Image.open(io.BytesIO(video.frame(topic, 0))) as image:
            width, height = image.size
        prepared[topic] = {}
        for key, shapes in frames.items():
            if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key or not 0 <= int(key) < video.topics[topic]["count"]:
                raise ValueError("候选帧号无效")
            if not isinstance(shapes, list):
                raise ValueError("候选帧必须是标注列表")
            result = []
            for index, shape in enumerate(shapes, 1):
                if not isinstance(shape, dict):
                    raise ValueError("候选标注格式无效")
                candidate = copy.deepcopy(shape)
                candidate["id"] = index
                if candidate.get("label") not in labels or not valid_shape(candidate, width, height):
                    raise ValueError("候选类别或几何无效")
                result.append(candidate)
            prepared[topic][key] = result
    return prepared


def merge_candidate_import(state, prepared):
    existing_truth = dict(state.get("truths_by_topic", {}))
    existing_candidates = {topic: dict(frames) for topic, frames in state.get("candidates_by_topic", {}).items()}
    current = state.get("topic")
    if current:
        existing_truth[current] = state.get("truth", {})
        existing_candidates[current] = dict(state.get("candidates", {}))
    for topic, frames in prepared.items():
        for key in frames:
            if key in existing_truth.get(topic, {}) or key in existing_candidates.get(topic, {}):
                raise ValueError("目标帧已有真值或候选，导入不会覆盖现有工作")
    for topic, frames in prepared.items():
        existing_candidates.setdefault(topic, {}).update(frames)
    return {"candidates_by_topic": existing_candidates,
            "candidates": existing_candidates.get(current, {})}
