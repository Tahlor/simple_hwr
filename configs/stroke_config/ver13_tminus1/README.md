# tminus1 - take previous prediction (or external data), feed it into RNN
#          - default: take previous, KD tree correct it, feed into next
#          - use DTW loss
    # Intent: 
    # Status
        Weird stalagmites at first, eventually diverges
        
        
# V2 - check if upsampling last is different from default64 CNN
    # Intent: make it a little faster
    # Status: doesn't make it much faster, diminished performance

# No KDTREE - don't correct previous prediction, just feed it in
    # Intent: make it a little faster


# previous    
    Status: converges to single output? possible problem graphing?
    
    ## previous - put correct GTs into RNN, use L1 loss on abs X
        # this appears to be dumb
        # should probably sum then do L1
        # VERY fast, since we can input the entire string to the RNN
    ## previous_REL - put correct GTs into RNN, use L1 loss on rel. x coords
