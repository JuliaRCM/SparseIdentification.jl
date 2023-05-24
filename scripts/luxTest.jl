
using Lux
using Images
using Random
using Optimisers

# Construct the relative file path
file_path = joinpath(@__DIR__, "luffy.jpeg")

# Load and process the image
img = load(file_path)
img_resized = Images.imresize(Gray.(img), (28, 28))
img_vec = Float32.(vec(img_resized))

# Define the encoder and decoder layers
enc_layer_1 = Lux.Dense(length(img_vec), 128, Lux.relu)
enc_layer_2 = Lux.Dense(128, 64, Lux.relu)
enc_layer_3 = Lux.Dense(64, 32, Lux.relu)
encoder = Lux.Chain(enc_layer_1, enc_layer_2, enc_layer_3)

dec_layer_1 = Lux.Dense(32, 64, Lux.relu)
dec_layer_2 = Lux.Dense(64, 128, Lux.relu)
dec_layer_3 = Lux.Dense(128, length(img_vec), Lux.relu)
decoder = Lux.Chain(dec_layer_1, dec_layer_2, dec_layer_3)

# Combine encoder and decoder into an autoencoder
autoencoder = Lux.Chain(encoder, decoder)

# specify the random number generator
rng = Random.default_rng()

# Lux setup parameters
ps, st = Lux.setup(rng, autoencoder)

# Define the loss function as mean squared error
loss(autoencoder, ps, st, X) = sum(abs2, autoencoder(X, ps, st)[1] .- X)

function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end

# Create the optimiser
opt_state = create_optimiser(ps)

# Specify the number of epochs
num_epochs = 100

# Train the autoencoder for the specified number of epochs
for epoch in 1:num_epochs
    # Compute the gradients and update the parameters
    # Lux.train!(loss, ps, [(img_vec, img_vec)], opt)
    loss_val = loss(autoencoder, ps, st, img_vec)
    gs = back((one(loss_val), nothing, nothing))[1]
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
end


# Pass the input through the autoencoder and calculate the loss
output = autoencoder(img_vec, ps, st)
error = loss(autoencoder, ps, st, img_vec)

# Print the output and error
println("Output:")
println(output)
println("Error: ", error)


# Convert the 1D vector back to a 2D image
img_resized = reshape(output[1], (28, 28))

# Convert the image back to grayscale
img_gray = Gray.(img_resized)

