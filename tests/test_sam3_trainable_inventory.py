import unittest

from tools.sam3_trainable_inventory import inventory_named_parameters, set_trainable_stage


class FakeParameter:
    def __init__(self):
        self.requires_grad = True


class FakeSam3:
    def __init__(self):
        names = [
            "backbone.vision_backbone.blocks.0.weight",
            "backbone.language_backbone.encoder.weight",
            "segmentation_head.mask_predictor.mask_embed.layers.0.weight",
            "segmentation_head.pixel_decoder.conv_layers.0.weight",
            "geometry_encoder.label_embed.weight",
            "geometry_encoder.points_direct_project.weight",
            "transformer.decoder.layers.0.self_attn.weight",
            "transformer.decoder.layers.1.self_attn.weight",
            "transformer.decoder.layers.2.self_attn.weight",
            "transformer.decoder.layers.3.self_attn.weight",
            "transformer.decoder.norm.weight",
        ]
        self.parameters = [(name, FakeParameter()) for name in names]

    def named_parameters(self):
        return iter(self.parameters)


class Sam3TrainableInventoryTests(unittest.TestCase):
    def test_inventory_uses_real_names_and_derives_upper_decoder_half(self):
        inventory = inventory_named_parameters(FakeSam3())
        self.assertEqual(inventory["decoder_layer_indices"], [0, 1, 2, 3])
        self.assertEqual(inventory["runtime_upper_decoder_indices"], [2, 3])
        self.assertEqual(inventory["groups"]["stage0_mask_embed"]["parameter_count"], 1)
        self.assertEqual(inventory["groups"]["stage1_full_segmentation_head"]["parameter_count"], 2)
        self.assertEqual(inventory["groups"]["stage2_geometry_prompt_encoder"]["parameter_count"], 2)
        self.assertEqual(inventory["adapter_or_lora_parameters"], [])
        self.assertFalse(inventory["stage0_is_final_upper_bound"])

    def test_each_stage_is_reversible_and_backbones_remain_frozen(self):
        model = FakeSam3()
        stage0 = set_trainable_stage(model, 0)
        self.assertEqual(stage0["trainable_names"], [
            "segmentation_head.mask_predictor.mask_embed.layers.0.weight"
        ])
        stage1 = set_trainable_stage(model, 1)
        self.assertIn("segmentation_head.pixel_decoder.conv_layers.0.weight",
                      stage1["trainable_names"])
        stage2 = set_trainable_stage(model, 2)
        self.assertIn("geometry_encoder.label_embed.weight", stage2["trainable_names"])
        stage3 = set_trainable_stage(model, 3)
        self.assertIn("transformer.decoder.layers.2.self_attn.weight",
                      stage3["trainable_names"])
        self.assertNotIn("transformer.decoder.layers.0.self_attn.weight",
                         stage3["trainable_names"])
        self.assertFalse(dict(model.parameters)[
            "backbone.vision_backbone.blocks.0.weight"].requires_grad)
        self.assertFalse(dict(model.parameters)[
            "backbone.language_backbone.encoder.weight"].requires_grad)


if __name__ == "__main__":
    unittest.main()
