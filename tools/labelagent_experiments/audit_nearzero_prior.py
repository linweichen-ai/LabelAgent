"""Score only complete frozen raw outputs with unchanged acceptance gates."""
import audit_soft_prior as audit
from run_nearzero_prior import OUT

if __name__=='__main__':
    audit.OUT=OUT
    audit.main()
