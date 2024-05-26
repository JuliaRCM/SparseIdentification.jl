"""
Generates the Lorenz system of equations.

# Arguments
- `y`: A vector of the state variables [x, y, z].
- `p`: A tuple of the parameters (σ, β, ρ).
- `t`: Time variable (not used in the equations but typically required for ODE solvers).

# Returns
- A vector `dy` representing the time derivatives of the state variables.

The Lorenz system is defined by the following equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
"""
function lorenz(y, p, t)
    σ, β, ρ = p
    dy = [
        σ*(y[2] - y[1]);
        y[1]*(ρ - y[3]) - y[2];
        y[1]*y[2] - β*y[3];
    ]
    return dy
end
