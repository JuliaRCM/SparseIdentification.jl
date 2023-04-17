" functions to generate hamiltonian function of variable z and 
order 3 of combinations with 2 dims for each variable "

_prod(a, b, c, arrs...) = a .* _prod(b, c, arrs...)
_prod(a, b) = a .* b
_prod(a) = a


"""
returns the number of required parameters taking into account many types of basis functions
"""
function calculate_nparams(nd, polyorder, trig_wave_num, diffs_power, trig_state_diffs)
    # binomial used to get the combination of polynomials till the highest order without repeat, e.g nparam = 34 for 3rd order, with z = q,p each of 2 dims
    # nd: total number of dims of all variable states
    nparam = binomial(nd + polyorder, polyorder) - 1

    if trig_wave_num > 0
        # first 2 in the product formula b/c the trig basis are sin and cos i.e. two basis functions
        nparam += 2 * trig_wave_num * nd
    end

    if abs(diffs_power) > 0
        # diffs power is the max power of the difference of states in the library of basis functions
        nparam += abs(diffs_power) * nd * (nd-1)
    end

    if abs(trig_state_diffs) > 0 
        # we add this b/c we also want to get the powers of sin and cos of the difference of states in the function library
        nparam += 2 * abs(trig_state_diffs) * nd * (nd-1)
    end

    return nparam
end
