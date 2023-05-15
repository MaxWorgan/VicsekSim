using DrWatson
quickactivate(@__DIR__)

using Flux.Data: DataLoader
using Flux, DataFrames, StatsBase,MLDataPattern, CUDA, PlotlyJS, LegolasFlux, CSV
using Wandb, Dates,Logging

## Start a new run, tracking hyperparameters in config
logger = WandbLogger(
   project = "Vicsek-VAE",
   name = "vicsek-training-$(now())",
   config = Dict(
      "η" => 0.00001,
      "batch_size" => 48,
      "data_set" => "data_test"
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

function vae_loss(encoder_μ, encoder_logvar, decoder, x)
    len = size(x)[end]
    @assert len != 0
    # Forward propagate through mean encoder and std encoders
    μ = encoder_μ(x)
    logσ = encoder_logvar(x)
    z = μ + gpu(randn(Float32, size(logσ))) .* exp.(logσ)
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
        Conv((9,), 200 => 2000, relu; pad=SamePad()),
        MaxPool((2,)),
        Conv((5,), 2000 => 1500, relu; pad=SamePad()),
        MaxPool((2,)),
        Conv((5,), 1500 => 750, relu; pad=SamePad()),
        MaxPool((3,)),
        Conv((3,), 750 => 250, relu; pad=SamePad()),
        Conv((3,), 250 => 25, relu; pad=SamePad()),
        Conv((3,), 25 => 10, relu; pad=SamePad()),
        Flux.flatten,
        Dense(50, 10, relu)
    )

    encoder_μ = Chain(encoder_features, Dense(10, 10))

    encoder_logvar = Chain(encoder_features, Dense(10, 10))

    decoder = Chain(
        Dense(10, 50, relu),
        (x -> reshape(x, 5, 10, :)),
        ConvTranspose((3,), 10 => 25, relu; pad=SamePad()),
        ConvTranspose((3,), 25 => 250, relu; pad=SamePad()),
        ConvTranspose((3,), 250 => 750, relu; pad=SamePad()),
        Upsample((3,)),
        ConvTranspose((5,), 750 => 1500, relu; pad=SamePad()),
        Upsample((2,)),
        ConvTranspose((5,), 1500 => 2000, relu; pad=SamePad()),
        Upsample((2,)),
        ConvTranspose((9,), 2000 => 200; pad=SamePad()),
    )
    return (encoder_μ, encoder_logvar, decoder)

end

function save_model(m, epoch, loss)
    model_row = LegolasFlux.ModelV1(; weights = fetch_weights(cpu(m)),architecture_version=1, loss=loss)
    write_model_row("1d_100_model-vae-$epoch-$loss.arrow", model_row)
end

function rearrange_1D(x)
    permutedims(cat(x..., dims=3), [2,1,3])
end



function train!(encoder_μ, encoder_logvar, decoder, train, validate, opt_enc_μ, opt_enc_logvar, opt_dec; num_epochs=100, dev=Flux.gpu)
    for e = 1:num_epochs
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
        for y in validate
            y = y |> dev
            validate_loss = vae_loss(encoder_μ, encoder_logvar, decoder, y)
            @info "metrics" validate_loss = validate_loss
        end
    end
end
 

function normalise(M) 
    min_m = minimum(M)
    max_m = maximum(M)
    return (M .- min_m) ./ (max_m - min_m)
end


function load_data(file_path, window_size)
    
    df           = DataFrame(CSV.File(file_path; types=Float32));
    flattened    = reduce(vcat, eachrow(Matrix(df[!, [:x, :y]])))

    windowed     = slidingwindow(reshape(flattened, (200, :)), window_size, stride=1)

    ts, vs       = splitobs(shuffleobs(windowed), 0.7)
    ts_length    = length(ts)
    vs_length    = length(vs)

    train_set    = permutedims(reshape(reduce(hcat, ts), (200,window_size,ts_length)), (2,1,3))
    validate_set = permutedims(reshape(reduce(hcat, vs), (200,window_size,vs_length)), (2,1,3))

    train_loader    = DataLoader(mapslices(normalise,train_set; dims=3); batchsize=48,shuffle=true)
    validate_loader = DataLoader(mapslices(normalise,validate_set; dims=3); batchsize=48,shuffle=true)

    (train_loader, validate_loader)

end



window_size = 60

(train_loader, validate_loader) = load_data("$(datadir())/sims/vicsek-eta0.15-r0.05-step0.02.csv", window_size)

num_epochs  = 250

encoder_μ, encoder_logvar, decoder = create_vae() |> gpu

train!(encoder_μ, encoder_logvar, decoder, train_loader, validate_loader, Flux.Optimise.ADAM(get_config(logger, "η")), num_epochs=num_epochs)

close(logger)
