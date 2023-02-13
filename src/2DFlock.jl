# The flock model illustrates how flocking behavior can emerge when each bird follows three simple rules:
#
# * maintain a minimum distance from other birds to avoid collision
# * fly towards the average position of neighbors
# * fly in the average direction of neighbors

# It is also available from the `Models` module as [`Models.flocking`](@ref).

# ## Defining the core structures

# We begin by calling the required packages and defining an agent type representing a bird.

using Agents, LinearAlgebra, Random, GLMakie, InteractiveDynamics

@agent Bird ContinuousAgent{2} begin
    θ::Float64
end

# The fields `id` and `pos`, which are required for agents on [`ContinuousSpace`](@ref),
# are part of the struct. The field `vel`, which is also added by
# using [`ContinuousAgent`](@ref) is required for using [`move_agent!`](@ref)
# in `ContinuousSpace` with a time-stepping method.
# `speed` defines how far the bird travels in the direction defined by `vel` per `step`.
# `separation` defines the minimum distance a bird must maintain from its neighbors.
# `visual_distance` refers to the distance a bird can see and defines a radius of neighboring birds.
# The contribution of each rule defined above receives an importance weight: `cohere_factor`
# is the importance of maintaining the average position of neighbors,
# `match_factor` is the importance of matching the average trajectory of neighboring birds,
# and `separate_factor` is the importance of maintaining the minimum
# distance from neighboring birds.

# The function `initialize_model` generates birds and returns a model object using default values.
function initialize_model(;
    n_birds   = 1000,
    step_size = 0.03,
    extent    = (10, 10),
    η         = 0.01,
    seed      = 12345,
    r         = 1.0
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
# according to the three rules defined above.
function agent_step!(bird, model)
    ## Obtain the ids of neighbors within the bird's visual distance
    neighbour_ids = nearby_ids(bird, model, model.r)
    ## Calculate behaviour properties based on neighbors
    mean_θ = [cos(bird.θ),sin(bird.θ)]
    for id in neighbour_ids
        mean_θ[1] += cos(model[id].θ)
        mean_θ[2] += sin(model[id].θ)
    end
    mean_θ = atan(mean_θ[2], mean_θ[1])

    noise = model.η * (rand(model.rng) * 2π - π)
    mean_θ += noise

    bird.vel = (cos(mean_θ),sin(mean_θ))

    move_agent!(bird, model, model.step_size)
end

# ## Plotting the flock

# The great thing about [`abmplot`](@ref) is its flexibility. We can incorporate the
# direction of the birds when plotting them, by making the "marker" function `am`
# create a `Polygon`: a triangle with same orientation as the bird's velocity.
# It is as simple as defining the following function:

const bird_polygon = Polygon(Point2f[(-0.5, -0.5), (1, 0), (-0.5, 0.5)])
function bird_marker(b::Bird)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    scale(rotate2D(bird_polygon, φ), 2)
end

# Where we have used the utility functions `scale` and `rotate2D` to act on a
# predefined polygon. We now give `bird_marker` to `abmplot`, and notice how
# the `as` keyword is meaningless when using polygons as markers.

model = initialize_model(;
    n_birds   = 300,
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



using Statistics:mean

adata = [(x -> x.vel, vs -> sqrt(sum(sum.(vs).^2) / length(model.agents)))]
alabels = ["order"]

parange = Dict(
    :step_size => 0.01:0.01:1.0,
    :η => 0.0:0.01:1.0,
    :r => 1:0.5:10.0
)

figure, _ = abmexploration(
    model;
    agent_step!,
    params = parange,
    as = 10,
    adata, alabels,

)
figure




