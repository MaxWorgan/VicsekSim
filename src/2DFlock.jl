# A very simple model with constant velocity; each bird/agent
# is updated to align to the average of its neighbours direction
# of travel. Additionally there is controllable noise added into
# the velocity calculation.
using DrWatson
quickactivate(@__DIR__)

    using Agents, LinearAlgebra, Random, InteractiveDynamics
    using Statistics
    using Graphs
    using Colors
    using ColorSchemes
    using DataStructures
    using CSV
    # using GLMakie
    using DataFrames

using Plotly
using StatsPlots


include("Measurements.jl")
include("ViscekModel.jl")


model = initialize_model(;
    n_birds=100,
    step_size=0.02,
    extent=(1,1),
    η=0.15,
    seed=12350,
    r=0.05)

mdata = [calculate_order]#, calculate_fractal_sevcik_graph_window]
mlabels = ["Order"]#, "Fractal Graph Window"]#, "Edit Distance"]

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
    as = 10,
    mdata, mlabels,
)

figure
model.properties

adata = [a -> a.pos[1], a->a.pos[2]]

a_Test, _ = run!(model, dummystep, model_step!, 1; adata )

agent_df, _ = run!(model, dummystep, model_step!, 100000; adata )

agent_df

CSV.write("$(datadir())/sims/vicsek-100-5.csv", agent_df; header=[:step, :id, :x, :y])

@df model_df StatsPlots.plot(:step, :calculate_fractal_sevcik_graph_window)

df = DataFrame(CSV.File("data/sims/vicsek-eta0.15-r0.05-step0.02.csv";types=Float32))

model_df

fig, _, obs = abmplot(model; ac = (x -> x.backbone))
fig

obs

fig

params = Dict(
    :η => collect(0.0:0.1:1.0),
    :n_birds=>100,
    :step_size=>0.02,
    :extent=>(2, 2),
    :seed=>12345,
    :r=>0.05
)

_, mdf = paramscan(params, initialize_model; model_step!, mdata, n=10000, parallel = true, showprogress=true)

mdf
palette(:gnuplot2)
using PlotlyJS
@df filter!(row -> row.step > 100, mdf) StatsPlots.plot(mdf.step, mdf.calculate_fractal_sevcik_graph_window, group=mdf.η)

# PlotlyJS.savefig(
PlotlyJS.plot(mdf, x=:step, y=:calculate_fractal_sevcik_graph_window,
    color=:η,
    linewidth=1,
    thickness_scaling=1,
    dpi=900,
    line=attr(width=0.5),
    colormap=cgrad(ColorSchemes.winter, 10)
)
    # , "plots/fractal_window2.png")


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