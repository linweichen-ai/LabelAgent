"""Neutral semantic controls; preserve exact model-bound response provenance."""
import sys
import review_soft_prior_controls as review
from run_nearzero_prior import OUT

if __name__=='__main__':
    review.OUT=OUT
    review.D=OUT/'semantic_controls'
    review.prepare() if sys.argv[1]=='prepare' else review.run()
