# TO DO -- Winter 2022

## Compute p(W)
## run in a regime closer to equilibrium such that the landscape reconstruction is really close
## try optimizing for diff # of points (1, 5, 10)
## step two: determine the form of the loss function needed to optimize the landscape

# Paper structure/ideas:

## A. JAX-MD can reproduce results that came before
## B. Present 'new' way of quantifying landscape errors (sampling at multiple points) and show that it is better -> what is the correct loss function to use?
## C. Iterative thing working -> need to test on (a) highly asymmetric landscapes (b) landscapes with intermediates

# Timeline:

## Formal check-in once per week
## 1 month: decide which of A-C is ready to write up


## Old to-dos:

## 1. Determine mean+std deviation over several (~10) estimates of the free energy difference via work-optimized, error-optimized, and linear protocols. X
## 2. Figure out memory issues -- try implementing checkpoint_scan X
## 3. Try 1000-5000 optimization steps X
## 4. Instead of equally distributing bins, try optimization for 10 bins in the folded well, 10 bins in the barrier region, then 10 bins in the unfolded well, and see if perhaps those landscapes can be "stitched" together afterward for a more accurate estimate. 
## 5. Is there hope for the iterative procedure? -- Oliver
## 6. Figure out why using the "wrong" stiffness in the reconstructions gives a better landscape --Zosia
## 7. Perform a more systematic grid search across simulation length and trap stiffness space, roughly quantifying the landscape error in each case (maybe sim of squared distance?),  to see how that error behaves as we tune the parameters (trajectory, protocol, landscape error) -- Zosia k_s, Oliver simulation length (k_s = 0.6), r0_final -- Zosia
## 8. how valid is the "dE doesn't affect the optimal protocol" assumption of Sivak & Crooks (2016) --Oliver
## 9. p(W) distributions! 
## landscape quality metric: align the folded well, measure discrepancies at every x, take the max, and that's your "bias"/"error"
