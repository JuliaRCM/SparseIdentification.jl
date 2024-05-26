"""
The `lorenz` function generates a Lorenz system of equations.

# Arguments
- `y`: state of the Lorenz system.
- `p`: parameters of the Lorenz system.
- `t`: time.

# Returns
- The derivative of the Lorenz system.
"""
function lorenz(y, p, t)
    sigma, beta, rho = p
    dy = [
    sigma*(y[2]-y[1]);
    y[1]*(rho-y[3])-y[2];
    y[1]*y[2]-beta*y[3];
    ]
    return dy
end
    
    