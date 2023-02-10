" functions to generate hamiltonian function of variable z and 
order 3 of combinations with 2 dims for each variable "

_prod(a, b, c, arrs...) = a .* _prod(b, c, arrs...)
_prod(a, b) = a .* b
_prod(a) = a


"""
returns the number of required parameters
depending on whether there are trig basis or not
"""
function calculate_nparams(d, polyorder, trig_wave_num)
    # binomial used to get the combination of polynomials till the highest order without repeat, e.g nparam = 34 for 3rd order, with z = q,p each of 2 dims
    nparam = binomial(2d + polyorder, polyorder) - 1

    if trig_wave_num > 0
        # first 2 in the product formula b/c the trig basis are sin and cos i.e. two basis functions
        # 2d: b/c the phase space is two variables p,q each with 2 dims
        nparam += 2 * trig_wave_num * 2d
    end

    return nparam
end
