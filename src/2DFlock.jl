# This is an implementation of the Vicsek model of flocking
#
# A very simple model with constant velocity; each bird/agent
# is updated to align to the average of its neighbours direction
# of travel. Additionally there is controllable noise added into
# the velocity calculation.

using Agents, LinearAlgebra, Random, GLMakie, InteractiveDynamics
using StaticArrays


@agent Bird ContinuousAgent{2} begin
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

    model = UnkillableABM(Bird, space2d; properties = properties, scheduler = Schedulers.fastest, rng)

    for _ in 1:n_birds
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        add_agent!(model, vel)
    end

    return model

end

function agent_step!(bird, model)
    ## Obtain the ids of neighbors within the bird's visual distance
    neighbour_ids = nearby_ids(bird, model, model.r)
    ## Calculate the mean velocity of neighbours
    mean_vel = [bird.vel...]
    for id in neighbour_ids
        mean_vel  += [model[id].vel...]
    end
    mean_θ = atan(mean_vel[2], mean_vel[1])
    # add some noise to the resulting angle
    noise = (rand(model.rng) * (2 * model.η) - model.η) 
    mean_θ += noise


    # set the velocity
    bird.vel = (cos(mean_θ),sin(mean_θ))
    
    move_agent!(bird, model, model.step_size)
end

model = initialize_model(;
    n_birds   = 2000,
    step_size = 0.03,
    extent    = (5, 5),
    η         = 0.2,
    seed      = 12345,
    r         = 0.1)

function calculate_order(model)
    velocities = map(a -> a.vel,model.agents)
    sum_vel    = reduce(.+, velocities)
    return (sum_vel[1]^2 + sum_vel[2]^2)^0.5 / length(model.agents)
end

mdata   = [calculate_order]
mlabels = ["order"]

parange = Dict(
    :step_size => 0.01:0.01:1.0,
    :η => 0.0:0.01:1.0,
    :r => 0.01:0.01:1.0,
)

figure, plot_data = abmexploration(
    model;
    agent_step!,
    params = parange,
    as = 10,
    mdata, mlabels,
)

figure

# abmvideo(
#     "flocking.mp4", model, agent_step!;
#     #am = bird_marker,
#     framerate = 30, frames = 300,
#     title = "Flocking"
# )