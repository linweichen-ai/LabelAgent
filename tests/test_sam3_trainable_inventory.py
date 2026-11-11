import unittest

from tools.sam3_trainable_inventory import inventory_named_parameters, set_trainable_stage


class FakeParameter:
    def __init__(self, count):
        self.requires_grad = True
        self.count = count

    def numel(self):
        return self.count


class FakeSam3:
    def __init__(self):
        specs = [
            ("backbone.vision_backbone.blocks.0.weight", 100),
            ("backbone.language_backbone.encoder.weight", 200),
            ("segmentation_head.mask_predictor.mask_embed.layers.0.weight", 30),
            ("segmentation_head.pixel_decoder.conv_layers.0.weight", 40),
            ("transformer.decoder.layers.0.self_attn.weight", 50),
            ("transformer.decoder.layers.1.self_attn.weight", 60),
        ]
        self.parameters = [(name, FakeParameter(count)) for name, count in specs]

    def named_parameters(self):
        return iter(self.parameters)


class Sam3TrainableInventoryTests(unittest.TestCase):
    def test_inventory_reports_exact_stage0_and_keeps_decoder_inventory_unauthorized(self):
        inventory = inventory_named_parameters(FakeSam3())
        self.assertEqual(inventory["scope"],
                         "inventory_only_not_training_authorization")
        stage0 = inventory["verified_stage0_scope"]
        self.assertEqual(stage0["names"], [
            "segmentation_head.mask_predictor.mask_embed.layers.0.weight"
        ])
        self.assertEqual(stage0["tensor_count"], 1)
        self.assertEqual(stage0["parameter_count"], 30)
        conditional = inventory["conditional_inventory_only"]
        self.assertFalse(conditional["authorized_for_training"])
        self.assertEqual(conditional["decoder_like"]["tensor_count"], 2)
        self.assertEqual(conditional["adapter_or_lora_like"]["names"], [])
        self.assertFalse(inventory["conditional_expansion_enabled"])
        self.assertFalse(inventory["full_model_training_enabled"])

    def test_only_stage0_can_be_enabled_and_everything_else_remains_frozen(self):
        model = FakeSam3()
        stage0 = set_trainable_stage(model, 0)
        self.assertEqual(stage0["trainable_names"], [
            "segmentation_head.mask_predictor.mask_embed.layers.0.weight"
        ])
        self.assertEqual(stage0["trainable_parameter_count"], 30)
        states = {name: parameter.requires_grad for name, parameter in model.parameters}
        self.assertTrue(states[
            "segmentation_head.mask_predictor.mask_embed.layers.0.weight"])
        self.assertFalse(states[
            "segmentation_head.pixel_decoder.conv_layers.0.weight"])
        self.assertFalse(states["transformer.decoder.layers.1.self_attn.weight"])
        self.assertFalse(states["backbone.vision_backbone.blocks.0.weight"])
        self.assertFalse(states["backbone.language_backbone.encoder.weight"])

    def test_broader_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only Stage-0 is authorized"):
            set_trainable_stage(FakeSam3(), 1)


if __name__ == "__main__":
    unittest.main()
