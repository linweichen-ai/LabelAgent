import sys,unittest
from pathlib import Path
here=Path(__file__).resolve()
for root in [here.parent]+[p/'tools/labelagent_experiments' for p in here.parents]:
 if (root/'semantic_subject_guard.py').exists():sys.path.insert(0,str(root));break
from semantic_subject_guard import compatible_pallet_subject
class Binding(unittest.TestCase):
 def test_secondary_pallet_mentions_are_not_primary_subject(self):
  for text in ['a gray concrete surface with two gray rectangular containers in the foreground and a blue pallet in the background','a person beside a wooden pallet','a plastic container on a pallet','an industrial vehicle carrying a blue pallet','a stack of crates containing a pallet','a black box and a blue pallet','a metal frame and a plastic pallet']:
   self.assertFalse(compatible_pallet_subject({'object':text}),text)
 def test_primary_pallet_with_background_objects_remains_allowed(self):
  for text in ['a white plastic pallet with gray containers in the background','a wooden pallet on concrete','a stack of blue plastic pallets','there is a metal skid in the foreground','in the center of the image, a wooden pallet with two feet','the main object is a white plastic pallet','a pallet is visible in the background','a black and blue plastic pallet','a stack of dark blue and white pallets','a wooden and plastic pallet with boxes behind it']:
   self.assertTrue(compatible_pallet_subject({'object':text}),text)
 def test_uncertainty_and_vehicle_vetoes_unchanged(self):
  for text in ['possibly a wooden pallet','a pallet jack carrying a pallet','a forklift carrying a pallet','a uniform gray surface','no pallet visible']:
   self.assertFalse(compatible_pallet_subject({'object':text}),text)
if __name__=='__main__':unittest.main()
