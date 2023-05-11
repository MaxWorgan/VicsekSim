# A very simple model with constant velocity; each bird/agent
# is updated to align to the average of its neighbours direction
# of travel. Additionally there is controllable noise added into
# the velocity calculation.
using Distributed
addprocs(8)

@everywhere begin
    using Agents, LinearAlgebra, Random, GLMakie, InteractiveDynamics
    using Statistics
    using Graphs
    using Colors
    using ColorSchemes
    using DataStructures
end

using Plotly
using StatsPlots


@everywhere include("Measurements.jl")
@everywhere include("ViscekModel.jl")


model = initialize_model(;
    n_birds=3000,
    step_size=0.15,
    extent=(50, 50),
    η=0.2,
    seed=12345,
    r=0.5)

mdata = [calculate_order, calculate_fractal_sevcik_graph_window]
mlabels = ["Order", "Fractal Graph Window"]#, "Edit Distance"]

parange = Dict(
    :step_size => 0.01:0.01:1.0,
    :η => 0.0:0.01:1.0,
    :r => 0.01:0.01:1.0,
)
# make a basic arrow

color_scheme = reverse(ColorSchemes.rainbow2);

figure, plot_data = abmexploration(
    model;
    dummystep,# agent_step!,
    model_step!,
    params=parange,
    ac = (x -> "0x" * hex(color_scheme[(atan(x.vel[2], x.vel[1]) + π) / 2π])),
    as = 5,
    mdata, mlabels,
)

figure
model.properties

_, model_df = run!(model, dummystep, model_step!, 3000; mdata)

@df model_df StatsPlots.plot(:step, :calculate_fractal_sevcik_graph_window)

using Plots
model_df

fig, _, obs = abmplot(model; ac = (x -> x.backbone))
fig

obs

fig


params = Dict(
    :η => collect(0.0:0.1:1.0)
)

_, mdf = paramscan(params, initialize_model; model_step!, mdata, n=3000, parallel = true, showprogress=true)

mdf
palette(:gnuplot2)
using PlotlyJS
@df filter!(row -> row.step > 100, mdf) StatsPlots.plot(mdf.step, mdf.calculate_fractal_sevcik_graph_window, group=mdf.η)

PlotlyJS.savefig(PlotlyJS.plot(mdf, x=:step, y=:calculate_fractal_sevcik_graph_window, color=:η), "plots/fractal_window.png")


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