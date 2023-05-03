# A very simple model with constant velocity; each bird/agent
# is updated to align to the average of its neighbours direction
# of travel. Additionally there is controllable noise added into
# the velocity calculation.

using Agents, LinearAlgebra, Random, GLMakie, InteractiveDynamics
using LempelZiv
using Statistics
using Graphs


@agent Particle ContinuousAgent{2} begin
    backbone::Bool
    number_of_connections::Int
end

Base.@kwdef mutable struct ParticleParameters
    r::Float64 = 0.1
    η::Float64 = 0.01
    step_size::Float64 = 0.03
    cumulative_order::Vector{Float64} = []
    graph::SimpleGraph = Graph(0)
end

# The function `initialize_model` generates birds and returns a model object using default values.
function initialize_model(;
    n_birds=3000,     # the number of birds/agents in the sim
    seed=12345,    # random seed for repeatable results
    step_size=0.15,     # the step_size of the implementation
    extent=(50, 50), # the size of the 'world'
    η=0.01,     # the amount of noise
    r=0.5       # the size of each birds neighbourhood
)
    rng = Random.MersenneTwister(seed)

    space2d = ContinuousSpace(extent; spacing=r / 1.5)

    properties = ParticleParameters(r=r, η=η, step_size=step_size, graph=Graph(n_birds))

    model = UnremovableABM(Particle, space2d; properties=properties, scheduler=Schedulers.fastest, rng)

    for _ in 1:n_birds
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
    add_agent!(model, vel, false,0)
    end

    return model

end

function model_step!(model)

    model.properties.graph = Graph(length(model.agents))

    for a in Schedulers.fastest(model)
        model.agents[a].backbone = false
        model.agents[a].number_of_connections = 0
        agent_step!(model.agents[a],model)
    end
    mst,w = boruvka_mst(model.graph)
    for i in mst
        model.agents[i.src].backbone = true
        model.agents[i.src].number_of_connections += 1
        model.agents[i.dst].backbone = true
        model.agents[i.dst].number_of_connections += 1
    end

end

heatmap

function agent_step!(particle, model)
    ## Obtain the ids of neighbors within the bird's visual distance
    neighbour_ids = nearby_ids(particle, model, model.r)
    ## Calculate the mean velocity of neighbours
    mean_vel = [particle.vel...]
    for id in neighbour_ids
        add_edge!(model.graph, particle.id, id)
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


function calculate_order(model)
    velocities = map(a -> a.vel, model.agents)
    sum_vel = reduce(.+, velocities)
    order = (sum_vel[1]^2 + sum_vel[2]^2)^0.5 / length(model.agents)
    push!(model.properties.cumulative_order, order) # eww
    return order
end

function calculate_susceptibility(model)
    mean_sq = mean(map(x -> x^2, model.properties.cumulative_order))
    sq_mean = mean(model.properties.cumulative_order)^2.0
    return 1 / length(model.agents) * (mean_sq - sq_mean)
    # return var(model.properties.cumulative_order)
end

# Add our own bitstring method to allow easy conversion of pos data
import Base.bitstring
function bitstring(t::Tuple{Float32,Float32})
    return bitstring(t[1]) * bitstring(t[2])
end

import Core.Float32
function Float32(t::Tuple{Float64,Float64})
    return (Float32(t[1]), Float32(t[2]))
end

function calculate_lz(model)
    binary_string = gridspace_to_binary(model.space.grid)
    return LempelZiv.lempel_ziv_complexity(binary_string)
end

function gridspace_to_binary(g::GridSpace)
    return mapreduce(cell_to_binary, *, g.stored_ids)
end

function cell_to_binary(v::Vector{Int64})
    return mapreduce(string, *, digits(length(v), base=3))
end

model = initialize_model(;
    n_birds=3000,
    step_size=0.15,
    extent=(50, 50),
    η=0.0,
    seed=12345,
    r=0.5)

mdata = [calculate_order, calculate_susceptibility]
mlabels = ["Order", "Susceptibility"]

parange = Dict(
    :step_size => 0.01:0.01:1.0,
    :η => 0.0:0.01:1.0,
    :r => 0.01:0.01:1.0,
)
# make a basic arrow
const particle_polygon = scale(Polygon(Point2f[(-0.5, -0.5), (1, 0), (-0.5, 0.5), (-0.5, 0.1), (-3.0, 0.1), (-3.0, -0.1), (-0.5, -0.1)]), 0.3)

# rotate the arrow based on the angle of the particle
function particle_marker(p::Particle)
    φ = atan(p.vel[2], p.vel[1])
    scale(rotate2D(particle_polygon, φ), 2)
end

heatarray = :number_of_connections
heatkwargs = (colorrange = (-20, 60), colormap = :thermal)
plotkwargs = (;
    ac = (x -> x.number_of_connections),
    as = 10,
    am = particle_marker,
    scatterkwargs = (strokewidth = 1.0,),
    heatarray, heatkwargs
)

color_scheme = reverse(ColorSchemes.grays)

figure, plot_data = abmexploration(
    model;
    dummystep,# agent_step!,
    model_step!,
    params=parange,
    ac = (x -> "0x" * hex(color_scheme[x.number_of_connections/20.0])),
    as = 5,
    opacity=0.2,
    markeralpha=0.5,
    blend=:multiply,
    # am = particle_marker,
    # scatterkwargs = (strokewidth = 1.0,),
    # heatarray, heatkwargs,
    # plotkwargs,
    # as=10,
    # ac= (x -> x.backbone),
    # am=particle_marker,
    mdata, mlabels,
)

figure

agent_df, model_df = run!(model, agent_step!, 10)

fig, _, obs = abmplot(model; ac = (x -> x.backbone))
fig

obs

fig


params = Dict(
    :η => collect(0.0:0.01:1.0)
)

_, mdf = paramscan(params, initialize_model; agent_step!, mdata, n=5000, showprogress=true)
test = @chain mdf begin
    @by(:η, :order = last(:calculate_order), :susceptibility = var(:calculate_order), :m_sus = std(:calculate_order))
end

@df test StatsPlots.plot(test.η, test.order; label="order",legend=:topleft)
@df test StatsPlots.plot!(twinx(), test.η, test.m_sus, color=:red, label="susceptibility")



## Lets do some MI stuff


# grab x pos and θ
# xs = map(b ->(atan(b.vel[2],b.vel[1]), b.pos[1]), model.agents)
# scatter(xs;
#     axis = (; title = "", xlabel = "θ", ylabel="x")
# )

# heatmap!(map(x -> x[1], xs), map(x -> x[2], xs))

# abmvideo(
#     "flocking.mp4", model, agent_step!;
#     am=particle_marker,
#     framerate=30, frames=100,
#     title="Flocking"
# )