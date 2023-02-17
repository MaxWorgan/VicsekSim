# This is an implementation of the Vicsek model of flocking
#
# A very simple model with constant velocity; each bird/agent
# is updated to align to the average of its neighbours direction
# of travel. Additionally there is controllable noise added into
# the velocity calculation.

using Agents, LinearAlgebra, Random, GLMakie, InteractiveDynamics

@agent Bird ContinuousAgent{2} begin
    θ::Float64 # the angle of the birds trajectory
end

# The function `initialize_model` generates birds and returns a model object using default values.
function initialize_model(;
    n_birds   = 1000,     # the number of birds/agents in the sim
    seed      = 12345,    # random seed for repeatable results
    step_size = 0.03,     # the step_size of the implementation
    extent    = (10, 10), # the size of the 'world'
    η         = 0.01,     # the amount of noise
    r         = 1.0       # the size of each birds neighbourhood
)
    rng = Random.MersenneTwister(seed)

    space2d = ContinuousSpace(extent; spacing = 0.1)

    properties = Dict(
        :r         => r,
        :η         => η,
        :step_size => step_size
    )

    model = AgentBasedModel(Bird, space2d; properties = properties, scheduler = Schedulers.fastest, rng)

    for _ in 1:n_birds
        rand_θ = rand(model.rng) * 2π .- π # rand vec between [-π,π]
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        add_agent!(model, vel, rand_θ)
    end

    return model
end

# ## Defining the agent_step!
# `agent_step!` is the primary function called for each step and computes velocity
function agent_step!(bird, model)
    ## Obtain the ids of neighbors within the bird's visual distance
    neighbour_ids = nearby_ids(bird, model, model.r)
    ## Calculate the mean velocity of neighbours
    mean_θ = [cos(bird.θ),sin(bird.θ)]
    for id in neighbour_ids
        mean_θ[1] += cos(model[id].θ)
        mean_θ[2] += sin(model[id].θ)
    end
    mean_θ = atan(mean_θ[2], mean_θ[1])
    # add some noise to the resulting angle
    noise = (rand(model.rng) * (2 * model.η) - model.η) 
    mean_θ += noise
    # set the velocity
    bird.vel = (cos(mean_θ),sin(mean_θ))
    
    move_agent!(bird, model, model.step_size)
end

model = initialize_model(;
    n_birds   = 1000,
    step_size = 0.03,
    extent    = (5, 5),
    η         = 0.1,
    seed      = 12345,
    r         = 1.0)

abmvideo(
    "flocking.mp4", model, agent_step!;
    #am = bird_marker,
    framerate = 30, frames = 300,
    title = "Flocking"
)

# The order
adata = [(x -> x.vel, vs -> sqrt(sum(sum.(vs).^2) / length(model.agents)))]
alabels = ["order"]

parange = Dict(
    :step_size => 0.01:0.01:1.0,
    :η => 0.0:0.01:1.0,
    :r => 1:0.5:10.0
)

figure, plot_data = abmexploration(
    model;
    agent_step!,
    params = parange,
    as = 10,
    adata, alabels,

)
figure