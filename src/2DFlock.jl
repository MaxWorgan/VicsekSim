# This is an implementation of the Vicsek model of flocking
#
# A very simple model with constant velocity; each bird/agent
# is updated to align to the average of its neighbours direction
# of travel. Additionally there is controllable noise added into
# the velocity calculation.

using Agents, LinearAlgebra, Random, GLMakie, InteractiveDynamics

@agent VicsekParticle ContinuousAgent{2} begin end

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

    model = AgentBasedModel(VicsekParticle, space2d; properties = properties, scheduler = Schedulers.fastest, rng)

    for _ in 1:n_birds
        vel = normalize((rand(model.rng, 2) * 2.0) .- 1.0)
        add_agent!(model, Tuple(vel))
    end

    return model

end

# ## Defining the agent_step!
# `agent_step!` is the primary function called for each step and computes velocity
function agent_step!(particle, model)
    ## Obtain the ids of neighbors within the radius 
    neighbour_ids = nearby_ids(particle, model, model.r)

    ## Calculate the mean velocity of neibouring particles 
    sum_vel = [particle.vel[1],particle.vel[2]]
    for id in neighbour_ids
        sum_vel += [model[id].vel[1],model[id].vel[2]]
    end

    angle = atan(sum_vel[2], sum_vel[1])
    # add some noise to the resulting angle
    noise = (rand(model.rng) * 2.0 .- 1) * model.η
    angle += noise

    mean_vel = [cos(angle), sin(angle)]

    # set the velocity
    particle.vel = Tuple(mean_vel)
    
    move_agent!(particle, model, model.step_size)
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