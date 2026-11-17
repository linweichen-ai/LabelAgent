import importlib.util
from pathlib import Path
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "benchmark_incremental_prefix_callback.py")
SPEC = importlib.util.spec_from_file_location(
    "benchmark_incremental_prefix_callback", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.tolist_calls = 0

    def tolist(self):
        self.tolist_calls += 1
        return list(self.values)

    def __getitem__(self, index):
        return Scalar(self.values[index])


class FakeTokenEnforcer:
    def __init__(self):
        self.sequences = []

    def get_allowed_tokens(self, sequence):
        self.sequences.append(list(sequence))
        return type("Allowed", (), {"allowed_tokens": [len(sequence)]})()


class IncrementalPrefixCallbackTests(unittest.TestCase):
    def test_reads_full_prefix_once_then_only_new_tokens(self):
        enforcer = FakeTokenEnforcer()
        callback = MODULE.GreedyIncrementalPrefixAllowedTokensFn(enforcer)
        first = FakeTensor([1, 2, 3])
        second = FakeTensor([1, 2, 3, 4])
        third = FakeTensor([1, 2, 3, 4, 5])

        self.assertEqual(callback(0, first), [3])
        self.assertEqual(callback(0, second), [4])
        self.assertEqual(callback(0, third), [5])

        self.assertEqual(first.tolist_calls, 1)
        self.assertEqual(second.tolist_calls, 0)
        self.assertEqual(third.tolist_calls, 0)
        self.assertEqual(enforcer.sequences,
                         [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
        self.assertEqual(callback.full_readbacks, 1)
        self.assertEqual(callback.incremental_readbacks, 2)

    def test_unexpected_length_falls_back_to_full_readback(self):
        enforcer = FakeTokenEnforcer()
        callback = MODULE.GreedyIncrementalPrefixAllowedTokensFn(enforcer)
        callback(0, FakeTensor([1, 2]))
        jumped = FakeTensor([9, 8, 7, 6])

        self.assertEqual(callback(0, jumped), [4])

        self.assertEqual(jumped.tolist_calls, 1)
        self.assertEqual(enforcer.sequences[-1], [9, 8, 7, 6])
        self.assertEqual(callback.full_readbacks, 2)

    def test_batch_rows_are_isolated(self):
        enforcer = FakeTokenEnforcer()
        callback = MODULE.GreedyIncrementalPrefixAllowedTokensFn(enforcer)

        callback(0, FakeTensor([1, 2]))
        callback(1, FakeTensor([7, 8, 9]))
        callback(0, FakeTensor([1, 2, 3]))

        self.assertEqual(enforcer.sequences,
                         [[1, 2], [7, 8, 9], [1, 2, 3]])
        self.assertEqual(callback.full_readbacks, 2)
        self.assertEqual(callback.incremental_readbacks, 1)

    def test_aligns_nested_required_order_to_valid_target(self):
        schema = {"type": "object", "required": ["first", "second"],
                  "properties": {
                      "first": {"type": "object", "required": ["a", "b"],
                                "properties": {"a": {}, "b": {}}},
                      "second": {"type": "string"}}}
        target = {"second": "value", "first": {"b": 2, "a": 1}}

        MODULE.align_required_order(schema, target)

        self.assertEqual(schema["required"], ["second", "first"])
        self.assertEqual(schema["properties"]["first"]["required"],
                         ["b", "a"])


if __name__ == "__main__":
    unittest.main()
