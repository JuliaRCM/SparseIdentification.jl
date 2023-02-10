
using Symbolics

# define the number of variables, q,p in this case gives 2 variables
const d = 2

# highest polynomial order to check
const order = 3 

# binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
const nparam = binomial(2d + order, order) - 1

# total number of dimensions, where each variable has 2 dimensions
n = 2d

# verification check for number of variables
1 + n + n*(n+1) ÷ 2 + n*(n+1)*(n+2) ÷ 6 - 1

# get variables p and q and store in variable z
@variables q[1:d]
@variables p[1:d]

# a has a size of 34
@variables a[1:nparam]

z = vcat(q,p)

# define empty variable H to store the hamiltonian
H = Num(0)


function hamiltonian_poly(z, order, inds...)
    ham = []

    if order == 0
        Num(1)
    elseif order == length(inds)
        ham = vcat(ham, _prod([z[i] for i in inds]...))
    else
        start_ind = length(inds) == 0 ? 1 : inds[end]
        for j in start_ind:length(z)
            ham = vcat(ham, hamiltonian_poly(z, order, inds..., j))
        end
    end

    return ham
end

function hamiltonian(z, a, order)
    ham = []

    for i in 1:order
        ham = vcat(ham, hamiltonian_poly(z, i))
    end

    sum(collect(a .* ham))
end

H = hamiltonian(z, a, order)

# code that builds a function that gives the hamiltonian function. The built function is shown below called: hamilGradient!
# hamilFunction = build_function(H, z, a)

# ˍ₋arg1 = z (p,q) of 2 dims each 
# a is of size 34. The function below only works for poly order = 3

# Generates the hamiltonian that has 3rd order, 2 variables p,q, each in 2 dims
function hamiltonianFunction(ˍ₋arg1, a)
    begin
        (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((*)((^)(ˍ₋arg1[4], 2), (getindex)(a, 14)), (*)((^)(ˍ₋arg1[1], 2), (getindex)(a, 5))), (*)((^)(ˍ₋arg1[2], 2), (getindex)(a, 9))), (*)((^)(ˍ₋arg1[3], 2), (getindex)(a, 12))), (*)((^)(ˍ₋arg1[1], 3), (getindex)(a, 15))), (*)((^)(ˍ₋arg1[2], 3), (getindex)(a, 25))), (*)((^)(ˍ₋arg1[3], 3), (getindex)(a, 31))), (*)((^)(ˍ₋arg1[4], 3), (getindex)(a, 34))), (*)((getindex)(a, 3), ˍ₋arg1[3])), (*)((getindex)(a, 2), ˍ₋arg1[2])), (*)((getindex)(a, 4), ˍ₋arg1[4])), (*)((getindex)(a, 1), ˍ₋arg1[1])), (*)((*)((^)(ˍ₋arg1[3], 2), (getindex)(a, 32)), ˍ₋arg1[4])), (*)((*)((^)(ˍ₋arg1[2], 2), (getindex)(a, 26)), ˍ₋arg1[3])), (*)((*)((^)(ˍ₋arg1[3], 2), (getindex)(a, 22)), ˍ₋arg1[1])), (*)((*)((^)(ˍ₋arg1[3], 2), (getindex)(a, 28)), ˍ₋arg1[2])), (*)((*)((^)(ˍ₋arg1[4], 2), (getindex)(a, 24)), ˍ₋arg1[1])), (*)((*)((^)(ˍ₋arg1[4], 2), (getindex)(a, 30)), ˍ₋arg1[2])), (*)((*)((^)(ˍ₋arg1[1], 2), (getindex)(a, 17)), ˍ₋arg1[3])), (*)((*)((^)(ˍ₋arg1[1], 2), (getindex)(a, 18)), ˍ₋arg1[4])), (*)((*)((^)(ˍ₋arg1[2], 2), (getindex)(a, 27)), ˍ₋arg1[4])), (*)((*)((getindex)(a, 13), ˍ₋arg1[3]), ˍ₋arg1[4])), (*)((*)((^)(ˍ₋arg1[1], 2), (getindex)(a, 16)), ˍ₋arg1[2])), (*)((*)((^)(ˍ₋arg1[2], 2), (getindex)(a, 19)), ˍ₋arg1[1])), (*)((*)((^)(ˍ₋arg1[4], 2), (getindex)(a, 33)), ˍ₋arg1[3])), (*)((*)((getindex)(a, 7), ˍ₋arg1[3]), ˍ₋arg1[1])), (*)((*)((getindex)(a, 10), ˍ₋arg1[3]), ˍ₋arg1[2])), (*)((*)((getindex)(a, 8), ˍ₋arg1[4]), ˍ₋arg1[1])), (*)((*)((getindex)(a, 11), ˍ₋arg1[4]), ˍ₋arg1[2])), (*)((*)((getindex)(a, 6), ˍ₋arg1[1]), ˍ₋arg1[2])), (*)((*)((*)((getindex)(a, 20), ˍ₋arg1[3]), ˍ₋arg1[1]), ˍ₋arg1[2])), (*)((*)((*)((getindex)(a, 21), ˍ₋arg1[4]), ˍ₋arg1[1]), ˍ₋arg1[2])), (*)((*)((*)((getindex)(a, 23), ˍ₋arg1[3]), ˍ₋arg1[4]), ˍ₋arg1[1])), (*)((*)((*)((getindex)(a, 29), ˍ₋arg1[3]), ˍ₋arg1[4]), ˍ₋arg1[2]))
    end
end


# gradient of the Hamiltonian 
Dz = Differential.(z)
∇H = [expand_derivatives(dz(H)) for dz in Dz]

# multiply the hamiltonian with J, the skew-symmetric matrix to get a gradient that is actually hamiltonian in form
f = vcat(∇H[d+1:2d], -∇H[1:d])

# build the graident function out of symbolics. The built function is shown below called: hamilGradient!
# fcode = build_function(f, z, a)[2]

# output of fcode used to get function hamilFunction below

################## ################## ################## ################## ################## 
################## ################## ################## ################## ################## 
# Note: The function hamilGradient! below only works for poly order = 3. 
# It must be rebuilt using symbolics to find functions of other orders 
################## ################## ################## ################## ################## 
################## ################## ################## ################## ################## 

# t is unused and is only present to comply with ODEProblem syntax
function hamilGradient!(ˍ₋out,ˍ₋arg1, a::AbstractVector{T}, t) where T
    begin
        begin
            @inbounds begin
                ˍ₋out[1] = (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((*)((^)((getindex)(ˍ₋arg1, 4), 2), (getindex)(a, 33)), (*)((^)((getindex)(ˍ₋arg1, 1), 2), (getindex)(a, 17))), (*)((^)((getindex)(ˍ₋arg1, 2), 2), (getindex)(a, 26))), (*)((getindex)(a, 13), (getindex)(ˍ₋arg1, 4))), (*)((getindex)(a, 7), (getindex)(ˍ₋arg1, 1))), (*)((getindex)(a, 10), (getindex)(ˍ₋arg1, 2))), (*)((*)(2, (getindex)(a, 12)), (getindex)(ˍ₋arg1, 3))), (*)((*)(3, (^)((getindex)(ˍ₋arg1, 3), 2)), (getindex)(a, 31))), (*)((*)((getindex)(a, 20), (getindex)(ˍ₋arg1, 1)), (getindex)(ˍ₋arg1, 2))), (*)((*)((getindex)(a, 23), (getindex)(ˍ₋arg1, 4)), (getindex)(ˍ₋arg1, 1))), (*)((*)((getindex)(a, 29), (getindex)(ˍ₋arg1, 4)), (getindex)(ˍ₋arg1, 2))), (*)((*)((*)(2, (getindex)(a, 22)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 1))), (*)((*)((*)(2, (getindex)(a, 32)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 4))), (*)((*)((*)(2, (getindex)(a, 28)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 2))), (getindex)(a, 3))
                ˍ₋out[2] = (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((*)((^)((getindex)(ˍ₋arg1, 1), 2), (getindex)(a, 18)), (*)((^)((getindex)(ˍ₋arg1, 3), 2), (getindex)(a, 32))), (*)((^)((getindex)(ˍ₋arg1, 2), 2), (getindex)(a, 27))), (*)((getindex)(a, 13), (getindex)(ˍ₋arg1, 3))), (*)((getindex)(a, 8), (getindex)(ˍ₋arg1, 1))), (*)((getindex)(a, 11), (getindex)(ˍ₋arg1, 2))), (*)((*)(3, (^)((getindex)(ˍ₋arg1, 4), 2)), (getindex)(a, 34))), (*)((*)(2, (getindex)(a, 14)), (getindex)(ˍ₋arg1, 4))), (*)((*)((getindex)(a, 21), (getindex)(ˍ₋arg1, 1)), (getindex)(ˍ₋arg1, 2))), (*)((*)((getindex)(a, 23), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 1))), (*)((*)((getindex)(a, 29), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 2))), (*)((*)((*)(2, (getindex)(a, 33)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 4))), (*)((*)((*)(2, (getindex)(a, 24)), (getindex)(ˍ₋arg1, 4)), (getindex)(ˍ₋arg1, 1))), (*)((*)((*)(2, (getindex)(a, 30)), (getindex)(ˍ₋arg1, 4)), (getindex)(ˍ₋arg1, 2))), (getindex)(a, 4))
                ˍ₋out[3] = (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((*)(-1, (getindex)(a, 1)), (*)((*)(-1, (^)((getindex)(ˍ₋arg1, 3), 2)), (getindex)(a, 22))), (*)((*)(-1, (^)((getindex)(ˍ₋arg1, 4), 2)), (getindex)(a, 24))), (*)((*)(-1, (^)((getindex)(ˍ₋arg1, 2), 2)), (getindex)(a, 19))), (*)((*)(-2, (getindex)(a, 5)), (getindex)(ˍ₋arg1, 1))), (*)((*)(-1, (getindex)(a, 7)), (getindex)(ˍ₋arg1, 3))), (*)((*)(-1, (getindex)(a, 8)), (getindex)(ˍ₋arg1, 4))), (*)((*)(-1, (getindex)(a, 6)), (getindex)(ˍ₋arg1, 2))), (*)((*)(-3, (^)((getindex)(ˍ₋arg1, 1), 2)), (getindex)(a, 15))), (*)((*)((*)(-2, (getindex)(a, 17)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 1))), (*)((*)((*)(-2, (getindex)(a, 18)), (getindex)(ˍ₋arg1, 4)), (getindex)(ˍ₋arg1, 1))), (*)((*)((*)(-2, (getindex)(a, 16)), (getindex)(ˍ₋arg1, 1)), (getindex)(ˍ₋arg1, 2))), (*)((*)((*)(-1, (getindex)(a, 20)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 2))), (*)((*)((*)(-1, (getindex)(a, 21)), (getindex)(ˍ₋arg1, 4)), (getindex)(ˍ₋arg1, 2))), (*)((*)((*)(-1, (getindex)(a, 23)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 4)))
                ˍ₋out[4] = (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((*)(-1, (getindex)(a, 2)), (*)((*)(-1, (^)((getindex)(ˍ₋arg1, 1), 2)), (getindex)(a, 16))), (*)((*)(-1, (^)((getindex)(ˍ₋arg1, 3), 2)), (getindex)(a, 28))), (*)((*)(-1, (^)((getindex)(ˍ₋arg1, 4), 2)), (getindex)(a, 30))), (*)((*)(-1, (getindex)(a, 10)), (getindex)(ˍ₋arg1, 3))), (*)((*)(-1, (getindex)(a, 6)), (getindex)(ˍ₋arg1, 1))), (*)((*)(-1, (getindex)(a, 11)), (getindex)(ˍ₋arg1, 4))), (*)((*)(-2, (getindex)(a, 9)), (getindex)(ˍ₋arg1, 2))), (*)((*)(-3, (^)((getindex)(ˍ₋arg1, 2), 2)), (getindex)(a, 25))), (*)((*)((*)(-2, (getindex)(a, 19)), (getindex)(ˍ₋arg1, 1)), (getindex)(ˍ₋arg1, 2))), (*)((*)((*)(-1, (getindex)(a, 20)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 1))), (*)((*)((*)(-1, (getindex)(a, 21)), (getindex)(ˍ₋arg1, 4)), (getindex)(ˍ₋arg1, 1))), (*)((*)((*)(-2, (getindex)(a, 26)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 2))), (*)((*)((*)(-2, (getindex)(a, 27)), (getindex)(ˍ₋arg1, 4)), (getindex)(ˍ₋arg1, 2))), (*)((*)((*)(-1, (getindex)(a, 29)), (getindex)(ˍ₋arg1, 3)), (getindex)(ˍ₋arg1, 4)))
                nothing
            end
        end
    end
    return ˍ₋out
end



# Below commented out code is only for testing that the function runs

# initial test function
# x₀ = [2., 0., 0., 0.]

# let a vector be ones initially of length 34 (b/c 34 is number of poly combinations for 2 variables, with 2 dims of highest order 3)
# a = ones(34)

# 2 dims each of p and q gives 4 variables
# out = zeros(4)

# any random value or vector or tspan works for (t) below. t is unused and is only present to comply with ODEProblem syntax
# t = 0

# output of hamiltonFunction stored in out
# HAM = hamiltonianFunction(x₀, a)

# output of hamilTest stored in out
# test = hamilGradient!(ˍ₋out,ˍ₋arg1, a, t)