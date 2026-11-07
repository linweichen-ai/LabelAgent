"""Keep neutral blank controls and exact response reuse on the new policy."""
import sys
import review_soft_prior_controls as review
from run_rgb_neighbors import OUT

if __name__=='__main__':
    review.OUT=OUT
    review.D=OUT/'semantic_controls'
    review.prepare() if sys.argv[1]=='prepare' else review.run()
