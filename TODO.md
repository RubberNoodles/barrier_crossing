# TO DO

## 1. Determine mean+std deviation over several (~10) estimates of the free energy difference via work-optimized, error-optimized, and linear protocols.
## 2. Figure out memory issues -- try implementing checkpoint_scan
## 3. Try 1000-5000 optimization steps
## 4. Instead of equally distributing bins, try optimization for 10 bins in the folded well, 10 bins in the barrier region, then 10 bins in the unfolded well, and see if perhaps those landscapes can be "stitched" together afterward for a more accurate estimate.
## 5. Is there hope for the iterative procedure?

# Paper structure/ideas:

## A. JAX-MD can reproduce results that came before
## B. Present 'new' way of quantifying landscape errors (sampling at multiple points) and show that it is better
## C. Iterative thing working

# Timeline:

## Formal check-in once per week
## 1 month: decide which of A-C is ready to write up
