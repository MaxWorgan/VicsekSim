using DrWatson
quickactivate(@__DIR__)

using Flux.Data: DataLoader
using Flux, DataFrames, StatsBase,MLDataPattern, CUDA, PlotlyJS, LegolasFlux, CSV
using Wandb, Dates,Logging

include("Measurements.jl")
## Start a new run, tracking hyperparameters in config
logger = WandbLogger(
   project = "Vicsek-VAE",
   name = "vicsek-training-$(now())",
   config = Dict(
      "η" => 0.0000001,
      "batch_size" => 48,
      "data_set" => "vicsek-medium",
   ),
)

global_logger(logger)

function gaussian_nll(x̂, logσ, x)
    return 0.5 * (@. ( (x - x̂) / exp(logσ))^2 + logσ + 0.5 * log2(pi))
end

function softclip(input, min_val)
    return min_val .+ NNlib.softplus(input - min_val)
end

function reconstruction_loss(x̂, x)
    logσ = log(sqrt(mean((x - x̂).^2)))
    logσ = softclip(logσ, -6)
    rec  = sum(gaussian_nll(x̂, logσ, x))
    return rec
end

function vae_loss(encoder_μ, encoder_logvar, decoder, x;dev=gpu)
    len = size(x)[end]
    @assert len != 0
    # Forward propagate through mean encoder and std encoders
    μ = encoder_μ(x)
    logσ = encoder_logvar(x)
    z = μ + dev(randn(Float32, size(logσ))) .* exp.(logσ)
    # Reconstruct from latent sample
    x̂ = decoder(z)
    
    kl = -0.5 * sum(@. 1 + logσ - μ^2 - exp(logσ) )

    rec = reconstruction_loss(x̂, x)

    @info "metrics" reconstruction_loss=rec kl=kl
    
    return rec + kl
 
end


function create_vae()

    # 60x200xn
    encoder_features = Chain(
        Conv((8,), 400 => 4000, relu; pad=SamePad()),
        MaxPool((2,)),
        Conv((8,), 4000 => 2000, relu; pad=SamePad()),
        MaxPool((2,)),
        Conv((4,), 2000 => 1000, relu; pad=SamePad()),
        MaxPool((3,)),
        Conv((4,), 1000 => 250, relu; pad=SamePad()),
        Conv((2,), 250 => 25, relu; pad=SamePad()),
        Conv((2,), 25 => 10, relu; pad=SamePad()),
        Flux.flatten,
        Dense(50, 10, relu)
    )

    encoder_μ = Chain(encoder_features, Dense(10, 10))

    encoder_logvar = Chain(encoder_features, Dense(10, 10))

    decoder = Chain(
        Dense(10, 50, relu),
        (x -> reshape(x, 5, 10, :)),
        ConvTranspose((2,), 10 => 25, relu; pad=SamePad()),
        ConvTranspose((2,), 25 => 250, relu; pad=SamePad()),
        ConvTranspose((4,), 250 => 1000, relu; pad=SamePad()),
        Upsample((3,)),
        ConvTranspose((4,), 1000 => 2000, relu; pad=SamePad()),
        Upsample((2,)),
        ConvTranspose((8,), 2000 => 4000, relu; pad=SamePad()),
        Upsample((2,)),
        ConvTranspose((8,), 4000 => 400; pad=SamePad()),
    )
    return (encoder_μ, encoder_logvar, decoder)

end

function save_model(m, name)
    model_row = LegolasFlux.ModelV1(; weights = fetch_weights(cpu(m)),architecture_version=1)
    write_model_row("$name.arrow", model_row)
end

function train!(encoder_μ, encoder_logvar, decoder, train, validate, opt_enc_μ, opt_enc_logvar, opt_dec; num_epochs=100, dev=Flux.gpu)
    for e in 1:num_epochs
        for x in train
            x = x |> dev
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = Flux.pullback(encoder_μ, encoder_logvar, decoder) do enc_μ, enc_logvar, dec
                vae_loss(enc_μ, enc_logvar, dec, x; dev=dev)
            end
            @info "metrics" train_loss = loss

            # Feed the back 1 to obtain the gradients and update the model parameters
            grad_enc_μ, grad_enc_logvar, grad_dec = back(1.0f0)

            Flux.update!(opt_enc_μ, encoder_μ, grad_enc_μ)
            Flux.update!(opt_enc_logvar, encoder_logvar, grad_enc_logvar)
            Flux.update!(opt_dec, decoder, grad_dec)

        end
        # for y in validate
        #     y = y |> dev
        #     validate_loss = vae_loss(encoder_μ, encoder_logvar, decoder, y)
        #     @info "metrics" validate_loss = validate_loss
        # end

        @info "metrics" epoch = e
    end
end

function load_data(file_path, window_size)
    
    df           = DataFrame(CSV.File(file_path; types=Float32))
    ndf          = select(df,:,[:x,:y] => ((x,y) -> calculate_toroidal_coords.(x,y)) => AsTable)
    flattened    = reduce(vcat, eachrow(Matrix(ndf[!, [:x1,:x2,:x3,:x4]])))
    windowed     = slidingwindow(reshape(flattened, (400, :)), window_size, stride=1)
    ts, vs       = splitobs(windowed, 0.7)
    ts_length    = length(ts)
    vs_length    = length(vs)
    train_set    = permutedims(reshape(reduce(hcat, ts), (400,window_size,ts_length)), (2,1,3))
    validate_set = permutedims(reshape(reduce(hcat, vs), (400,window_size,vs_length)), (2,1,3))
    train_loader = DataLoader(train_set; batchsize=48,shuffle=true)
    validate_loader = DataLoader(validate_set; batchsize=48,shuffle=true)

    (train_loader, validate_loader)

end

window_size = 60

(train_loader, validate_loader) = load_data("$(datadir())/sims/$(get_config(logger, "data_set")).csv", window_size)

num_epochs  = 100

encoder_μ, encoder_logvar, decoder = create_vae() |> gpu

# ADAM optimizer
η = get_config(logger, "η")
opt_enc_μ = Flux.setup(Adam(η), encoder_μ)
opt_enc_logvar = Flux.setup(Adam(η), encoder_logvar)
opt_dec = Flux.setup(Adam(η), decoder)

train!(encoder_μ, encoder_logvar, decoder, train_loader, validate_loader, opt_enc_μ,opt_enc_logvar,opt_dec, num_epochs=num_epochs)

first(train_loader)

close(logger)

save_model(Chain(encoder_μ, encoder_logvar, decoder), "vicsek-model6")

file_path = "$(datadir())/sims/$(get_config(logger, "data_set")).csv"





x = first(train_loader)

plot(x[1,:,1];seriestype=:scatter)


z = mapslices(x -> reverseProjectFromCliffordTorus(x[1],x[2],x[3]), y; dims=1)

y = reshape(x[60,:,1], (3,100))
maptest = Iterators.map(z -> reverseProjectFromCliffordTorus(z[1],z[3],z[3]),eachcol(y)) 
xs = Iterators.map(z -> z[1], maptest) |> collect
ys = Iterators.map(z -> z[2], maptest) |> collect

plot(
    scatter(x=xs,y=ys, mode="markers"),
    Layout(xaxis_range=[0,1], yaxis_range=[0,1])
    )

plot(
    scatter(x = df[101:200,:x],
            y = df[101:200,:y],
            mode="markers"),
    Layout(xaxis_range=[0,1], yaxis_range=[0,1])
)

