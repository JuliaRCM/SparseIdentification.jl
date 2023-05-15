
using Lux
using Random

# Encoder layers
enc_layer_1 = Dense(784, 128, Lux.relu)
enc_layer_2 = Dense(128, 64, Lux.relu)
enc_layer_3 = Dense(64, 32, Lux.relu)

# Decoder layers
dec_layer_1 = Dense(32, 64, Lux.relu)
dec_layer_2 = Dense(64, 128, Lux.relu)
dec_layer_3 = Dense(128, 784, Lux.sigmoid)

# Define encoder and decoder networks
encoder = Chain(FlattenLayer(), enc_layer_1, enc_layer_2, enc_layer_3)
decoder = Chain(dec_layer_1, dec_layer_2, dec_layer_3)

# Combine encoder and decoder networks
autoencoder = Chain(encoder, decoder)

# Define loss function as mean squared error
function autoencoder_loss(x, y)
    return sum((autoencoder(x) - y).^2)
end


rng = Random.default_rng()
Random.seed!(rng, 0)
Random.TaskLocalRNG()




using Images

# Load image
img = load("/home/nigelbrucekhan/University/SparseIdentification.jl/scripts/luffy.jpeg")

# Convert to grayscale and resize to 28x28
img_gray = Gray.(img)
img_resized = Images.imresize(img_gray, (28, 28))

# Convert to a 1D array
x = reshape(img_resized, 28*28)

# Convert the image to a 1D vector of floats
img_vec = Float32.(vec(x))

ps, st = Lux.setup(rng, encoder)
autoencoder(img_vec, ps, st)














# Pass through autoencoder
y = autoencoder_loss(img_vec,img_vec)
