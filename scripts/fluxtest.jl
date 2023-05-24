
using Images
using Flux

# Construct the relative file path
file_path = joinpath(@__DIR__, "luffy.jpeg")

# Load and process the image
img = load(file_path)
img_resized = Images.imresize(Gray.(img), (28, 28))
img_vec = Float32.(vec(img_resized))

# Define the encoder and decoder layers
enc_layer_1 = Dense(784, 640, tanh)
enc_layer_2 = Dense(640, 384, tanh)
enc_layer_3 = Dense(384, 256, tanh)
enc_layer_4 = Dense(256, 128, tanh)
enc_layer_5 = Dense(128, 64, tanh)
enc_layer_6 = Dense(64, 32, tanh)
encoder = Chain(enc_layer_1, enc_layer_2, enc_layer_3, enc_layer_4, enc_layer_5, enc_layer_6)

dec_layer_1 = Dense(32, 64, sigmoid)
dec_layer_2 = Dense(64, 128, sigmoid)
dec_layer_3 = Dense(128, 256, sigmoid)
dec_layer_4 = Dense(256, 384, sigmoid)
dec_layer_5 = Dense(384, 640, sigmoid)
dec_layer_6 = Dense(640, 784, sigmoid)
decoder = Chain(dec_layer_1, dec_layer_2, dec_layer_3, dec_layer_4, dec_layer_5, dec_layer_6)

# Combine encoder and decoder into an autoencoder
autoencoder = Chain(encoder, decoder)

# Define the loss function as mean squared error
loss(x) = Flux.mse(autoencoder(x))

# Get the parameters of the autoencoder
ps = Flux.params(autoencoder)

opt = ADAM()  # Create an instance of ADAM optimizer

# Specify the number of epochs
num_epochs = 200

# Train the autoencoder for the specified number of epochs
for epoch in 1:num_epochs
    # Compute the gradients and update the parameters
    Flux.train!(loss, ps, [(img_vec, img_vec)], opt)
end

# Pass the input through the autoencoder and calculate the loss
output = autoencoder(img_vec)
error = loss(img_vec, img_vec)

# Print the output and error
println("Output:")
println(output)
println("Error: ", error)


# Convert the 1D vector back to a 2D image
img_resized = reshape(output, (28, 28))

# Convert the image back to grayscale
img_gray = Gray.(img_resized)