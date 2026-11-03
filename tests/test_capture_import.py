import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location("capture_import", Path(__file__).resolve().parents[1] / "tools/import_capture.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureImportTests(unittest.TestCase):
    def test_rejects_traversal_and_windows_drive_members_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("../outside", "folder/../../outside", "C:/outside", "/outside", "safe:stream", "folder./file"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    MODULE.safe_member(value, root)
            self.assertEqual(MODULE.safe_member("cameras/camera_0/123.png", root), root / "cameras/camera_0/123.png")

    def test_unsafe_archive_cannot_extract_an_earlier_safe_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("safe.txt", "safe")
                source.writestr("../outside", "unsafe")
            with self.assertRaises(ValueError):
                MODULE.extract(archive, root / "output")
            self.assertFalse((root / "output/safe.txt").exists())

    def test_truncated_pcd_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.pcd"
            path.write_bytes(b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n" + bytes(12))
            with self.assertRaises(ValueError):
                MODULE.read_pcd(path)

    def test_point_cloud_descriptor_preserves_payload_and_timestamp(self):
        descriptor, cls = MODULE.pointcloud_schema()
        source = cls(point_stride=12, data=bytes(24))
        source.timestamp.seconds, source.timestamp.nanos = 123, 456
        source.fields.add(name="x", offset=0, type=7)
        copy = cls.FromString(source.SerializeToString())
        self.assertTrue(descriptor)
        self.assertEqual((copy.timestamp.seconds, copy.timestamp.nanos), (123, 456))
        self.assertEqual(copy.data, bytes(24))


if __name__ == "__main__":
    unittest.main()
