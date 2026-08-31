
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


# gradient of the Hamiltonian
Dz = Differential.(z)
∇H = [expand_derivatives(dz(H)) for dz in Dz]

# multiply the hamiltonian with J, the skew-symmetric matrix to get a gradient that is actually hamiltonian in form
f = vcat(∇H[d+1:2d], -∇H[1:d])

# build the graident function out of symbolics. The built function is shown below called: hamilGradient!
# fcode = build_function(f, z, a)[2]

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
