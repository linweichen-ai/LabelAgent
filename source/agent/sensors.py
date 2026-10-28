"""Topic discovery and bounded point-cloud snapshots for the MCAP workbench."""
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def message_time(data, message):
    stamp = getattr(data, 'timestamp', None) or getattr(getattr(data, 'header', None), 'stamp', None)
    if stamp is not None:
        seconds = getattr(stamp, 'seconds', getattr(stamp, 'sec', 0))
        nanos = getattr(stamp, 'nanos', getattr(stamp, 'nanosec', 0))
        if seconds or nanos:
            return seconds * 1000000000 + nanos
    return message.log_time


def catalog(path):
    result = {}
    with path.open('rb') as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError('此 MCAP 缺少消息索引，请先添加 MCAP summary 索引')
        origin = summary.statistics.message_start_time
        for channel in summary.channels.values():
            schema = summary.schemas.get(channel.schema_id)
            name = schema.name if schema else ''
            lower = name.lower()
            kind = 'pointcloud' if 'pointcloud' in lower else 'image' if any(x in lower for x in ('compressedvideo', 'compressedimage', 'rawimage', 'sensor_msgs/msg/image', 'sensor_msgs.image')) else 'other'
            result[channel.topic] = dict(type=kind, schema=name, encoding=channel.message_encoding,
                count=summary.statistics.channel_message_counts.get(channel.id, 0))
        factory = DecoderFactory()
        for topic, info in result.items():
            if info['type'] == 'other':
                continue
            first = next(reader.iter_messages(topics=[topic]), None)
            if first:
                schema, channel, message = first
                decoder = factory.decoder_for(channel.message_encoding, schema)
                data = decoder(message.data) if decoder else None
                info['offset'] = (message_time(data, message) - origin) / 1e9
                info['log_offset'] = (message.log_time - origin) / 1e9
        duration = (summary.statistics.message_end_time - origin) / 1e9
    return result, origin, duration


# Foxglove PackedElementField numeric types; ROS PointField uses a different enum.
FOX_TYPES = {1: 'u1', 2: 'i1', 3: '<u2', 4: '<i2', 5: '<u4', 6: '<i4', 7: '<f4', 8: '<f8'}
ROS_TYPES = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}


def unpack_cloud(data):
    fox = hasattr(data, 'point_stride')
    stride = getattr(data, 'point_stride', 0) if fox else getattr(data, 'point_step', 0)
    if not stride:
        raise ValueError('点云缺少 point_stride / point_step')
    fields = {field.name: field for field in data.fields}
    if not all(axis in fields for axis in ('x', 'y', 'z')):
        raise ValueError('点云缺少 x、y、z 字段')
    payload = bytes(data.data)
    if not fox and getattr(data, 'row_step', 0) > data.width * stride:
        payload = b''.join(payload[y*data.row_step:y*data.row_step+data.width*stride] for y in range(data.height))
    count = len(payload) // stride
    values = []
    for name in ('x', 'y', 'z', next((x for x in ('intensity', 'reflectivity') if x in fields), '')):
        if not name:
            values.append(np.ones(count, dtype=np.float32))
            continue
        field = fields[name]
        numeric_type = field.type if fox else field.datatype
        dtype = (FOX_TYPES if fox else ROS_TYPES).get(numeric_type)
        if dtype is None:
            raise ValueError(f'不支持的点字段类型：{numeric_type}')
        if not fox and getattr(data, 'is_bigendian', False):
            dtype = dtype.replace('<', '>')
        if field.offset + np.dtype(dtype).itemsize > stride:
            raise ValueError('点字段超出步长范围')
        values.append(np.ndarray((count,), dtype=dtype, buffer=payload, offset=field.offset, strides=(stride,)))
    points = np.column_stack(values).astype(np.float32)
    points = points[np.isfinite(points).all(axis=1) & (np.linalg.norm(points[:, :3], axis=1) > 0.001)]
    if fox and hasattr(data, 'pose'):
        q = data.pose.orientation
        v = np.array([q.x, q.y, q.z], dtype=np.float32)
        norm = np.dot(v, v) + q.w*q.w
        if norm > 0:
            v, w = v / np.sqrt(norm), q.w / np.sqrt(norm)
            xyz = points[:, :3]
            cross = 2 * np.cross(v, xyz)
            points[:, :3] = xyz + w*cross + np.cross(v, cross)
        position = data.pose.position
        points[:, :3] += [position.x, position.y, position.z]
    return points


def cloud_snapshot(video, topic, seconds, window=0.1, limit=50000):
    if topic not in video.catalog or video.catalog[topic]['type'] != 'pointcloud':
        raise ValueError('请选择点云话题')
    # Query by log_time, compensating for the first capture/log timestamp offset.
    info = video.catalog[topic]
    capture_end = video.origin_ns + int(seconds * 1e9)
    log_end = capture_end + int((info.get('log_offset', 0)-info.get('offset', 0))*1e9)
    start = log_end - int(window*1e9)
    arrays = []
    messages = 0
    factory = DecoderFactory()
    with video.path.open('rb') as stream:
        for schema, channel, message in make_reader(stream).iter_messages(topics=[topic], start_time=max(video.origin_ns, start), end_time=log_end+1):
            decoder = factory.decoder_for(channel.message_encoding, schema)
            if decoder is None:
                raise ValueError('此点云消息编码尚不支持，需要 protobuf')
            data = decoder(message.data)
            arrays.append(unpack_cloud(data))
            messages += 1
            if messages >= 2000:
                break
    points = np.concatenate(arrays) if arrays else np.empty((0, 4), dtype=np.float32)
    original_count = len(points)
    if len(points) > limit:
        points = points[np.linspace(0, len(points)-1, limit, dtype=int)]
    return points.astype('<f4').tobytes(), original_count, messages
