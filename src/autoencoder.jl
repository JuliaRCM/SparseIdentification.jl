using Flux

function build_network_layers(input, input_dim, output_dim, widths, activation, name)
    """
    Construct one portion of the network (either encoder or decoder).
    Arguments:
        input - 2D Flux array, input to the network (shape is [?,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Flux function to be used as the activation function at each layer
        name - String, prefix to be used in naming the Flux parameters
    Returns:
        input - Flux array, output of the network layers (shape is [?,output_dim])
        weights - List of Flux arrays containing the network weights
        biases - List of Flux arrays containing the network biases
    """
    weights = []
    biases = []
    last_width = input_dim
    for (i, n_units) in enumerate(widths)
        W = param(name*"_W"*string(i), Flux.glorot_uniform(last_width, n_units))
        b = param(name*"_b"*string(i), zeros(n_units))
        input = input * W .+ b
        if activation !== nothing
            input = activation(input)
        end
        last_width = n_units
        push!(weights, W)
        push!(biases, b)
    end
    W = param(name*"_W"*string(length(widths)), Flux.glorot_uniform(last_width, output_dim))
    b = param(name*"_b"*string(length(widths)), zeros(output_dim))
    input = input * W .+ b
    push!(weights, W)
    push!(biases, b)
    return input, weights, biases
end
