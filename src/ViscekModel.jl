@agent Particle ContinuousAgent{2} begin
    
end

const GraphWindow = CircularBuffer{Vector{Float64}}

Base.@kwdef mutable struct ParticleParameters
    r::Float64 = 0.1 # neibourhood radius
    η::Float64 = 0.01 # noise
    step_size::Float64 = 0.03
    graph::Graph = Graph(0)
    graph_window::GraphWindow = GraphWindow(60)
    # prev_graph::SimpleGraph = Graph(0)
end

# The function `initialize_model` generates birds and returns a model object using default values.
function initialize_model(;
    n_birds=300,     # the number of birds/agents in the sim
    seed=12345,    # random seed for repeatable results
    step_size=0.15,     # the step_size of the implementation
    extent=(5, 5), # the size of the 'world'
    η=0.01,     # the amount of noise
    r=0.2       # the size of each birds neighbourhood
)
    rng = Random.MersenneTwister(seed)

    space2d = ContinuousSpace(extent; spacing=r / 2)

    properties = ParticleParameters(r=r, η=η, step_size=step_size, graph=Graph(n_birds))

    model = UnremovableABM(Particle, space2d; properties=properties, scheduler=Schedulers.fastest, rng)

    for _ in 1:n_birds
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        add_agent!(model, vel)
    end

    return model

end

function model_step!(model)

    # model.properties.prev_graph = copy(model.properties.graph)
    #push!(model.properties.graph_window, flatten_graph(model.properties.graph))
    # model.properties.graph = SimpleGraph(length(model.agents))
    for a in Schedulers.fastest(model)
        agent_step!(model.agents[a],model)
    end

end


function agent_step!(particle, model)
    ## Obtain the ids of neighbors within the bird's visual distance
    neighbour_ids = nearby_ids(particle, model, model.r)
    ## Calculate the mean velocity of neighbours
    mean_vel = [particle.vel...]
    for id in neighbour_ids
    #    add_edge!(model.graph, particle.id, id)
        mean_vel += [model[id].vel...]
    end
    mean_θ = atan(mean_vel[2], mean_vel[1])
    # add some noise to the resulting angle
    noise = (rand(model.rng) * (2 * model.η) - model.η)
    mean_θ += noise
    # set the velocity
    particle.vel = (cos(mean_θ), sin(mean_θ))

    move_agent!(particle, model, model.step_size)
end


