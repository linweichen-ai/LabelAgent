"""Original full-native-mask gates, only after cue/caption/cache plans freeze."""
import audit_soft_prior as audit
from run_material_anchor import OUT

if __name__=='__main__':
    audit.OUT=OUT
    audit.main()
