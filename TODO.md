# TO DO

## 1. Determine mean+std deviation over several (~10) estimates of the free energy difference via work-optimized, error-optimized, and linear protocols.
## 2. Figure out memory issues -- try implementing checkpoint_scan
## 3. Try 1000-5000 optimization steps
## 4. Instead of equally distributing bins, try optimization for 10 bins in the folded well, 10 bins in the barrier region, then 10 bins in the unfolded well, and see if perhaps those landscapes can be "stitched" together afterward for a more accurate estimate.
## 5. Is there hope for the iterative procedure?
## 6. Figure out why using the "wrong" stiffness in the reconstructions gives a better landscape
## 7. Perform a more systematic grid search across simulation length and trap stiffness space, roughly quantifying the landscape error in each case (maybe sim of squared distance?),  to see how that error behaves as we tune the parameters
## 8. how valid is the "dE doesn't affect the optimal protocol" assumption of Sivak & Crooks (2016)
## 9. p(W) distributions!

# Paper structure/ideas:

## A. JAX-MD can reproduce results that came before
## B. Present 'new' way of quantifying landscape errors (sampling at multiple points) and show that it is better
## C. Iterative thing working

# Timeline:

## Formal check-in once per week
## 1 month: decide which of A-C is ready to write up
