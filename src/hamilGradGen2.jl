
using Symbolics


" returns a function that can build the gradient of the hamiltonian "

function hamilGrad_func_builder(d, usesine, polyorder, nparam, trig_wave_num)
    
    # symbolic variables
    @variables a[1:nparam]
    @variables q[1:d]
    @variables p[1:d]
    z = vcat(q,p)

    # usesine: whether to add trig basis or not
    if usesine == true

        # gives derivative of the hamiltonian, but not the skew-symmetric true one
        Dz = Differential.(z)
        ∇H_add_trig = [expand_derivatives(dz(hamil_trig(z, a, polyorder, trig_wave_num))) for dz in Dz]

        # line below makes the vector into a hamiltonian vector field by multiplying with the skew-symmetric matrix
        ∇H_trig = vcat(∇H_add_trig[d+1:2d], -∇H_add_trig[1:d])

        # builds a function that calculates Hamiltonian gradient and converts the function to a native Julia function
        ∇H_eval = eval(build_function(∇H_trig, z, a)[2])

        return ∇H_eval

    elseif usesine == false

        # gives derivative of the hamiltonian, but not the skew-symmetric true one
        Dz = Differential.(z)
        f = [expand_derivatives(dz(hamiltonian(z, a, polyorder))) for dz in Dz]

        # line below makes the vector into a hamiltonian by multiplying with the skew-symmetric matrix
        ∇H = vcat(f[d+1:2d], -f[1:d])

        # builds a function that calculates Hamiltonian gradient and converts the function to a native Julia function
        ∇H_eval = eval(build_function(∇H, z, a)[2])
        
        return ∇H_eval

    end

end