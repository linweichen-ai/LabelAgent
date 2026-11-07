"""Apply the unchanged native-bitmap evaluator after layer prompts freeze."""
import audit_soft_prior as audit
from run_rgb_neighbors import OUT

if __name__=='__main__':
    audit.OUT=OUT
    audit.main()
