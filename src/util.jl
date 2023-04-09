" functions to generate hamiltonian function of variable z and 
order 3 of combinations with 2 dims for each variable "

_prod(a, b, c, arrs...) = a .* _prod(b, c, arrs...)
_prod(a, b) = a .* b
_prod(a) = a


"""
returns the number of required parameters
depending on whether there are trig basis or not
"""
function calculate_nparams(nd, polyorder, trig_wave_num, diffs_power)
    # binomial used to get the combination of polynomials till the highest order without repeat, e.g nparam = 34 for 3rd order, with z = q,p each of 2 dims
    #TODO: ask prof if this is correct: changed 2d to nd in all places, nd: total number to dims of all variable states
    # nd: total number of dims of all variable states
    nparam = binomial(nd + polyorder, polyorder) - 1

    if trig_wave_num > 0
        # first 2 in the product formula b/c the trig basis are sin and cos i.e. two basis functions
        nparam += 2 * trig_wave_num * nd
    end

    if diffs_power > 0
        # diffs power is the 
        nparam += diffs_power * nd * (nd-1)
    end

    return nparam
end