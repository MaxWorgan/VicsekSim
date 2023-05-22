
using LempelZiv
using Graphs
using Agents

include("Util.jl")

function calculate_order(model)
    velocities = map(a -> a.vel, model.agents)
    sum_vel = reduce(.+, velocities)
    order = (sum_vel[1]^2 + sum_vel[2]^2)^0.5 / length(model.agents)
    # push!(model.properties.cumulative_order, order) # eww
    return order
end

function calculate_susceptibility(model)
    mean_sq = mean(map(x -> x^2, model.properties.cumulative_order))
    sq_mean = mean(model.properties.cumulative_order)^2.0
    return 1 / length(model.agents) * (mean_sq - sq_mean)
    # return var(model.properties.cumulative_order)
end

function calculate_edit_distance(model)
    return edit_distance(model.properties.prev_graph, model.properties.graph)[1]
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

function calculate_fractal_sevcik_graph_window(model)
    # 1. Normalize the signal (new range to [0, 1])
    if model.properties.graph_window.length < model.properties.graph_window.capacity
        return 0.0
    end
    y⭐ = reduce(vcat, model.properties.graph_window; init=Vector{Float64}())

    n = length(y⭐)
    
    # 2. Derive x* and y* (y* is actually the normalized signal)
    x⭐ = LinRange(0.0, 1.0, n)

    # 3. Compute L (because we use np.diff, hence n-1 below)
    L = sum(sqrt.(diff(y⭐) .^ 2 + diff(x⭐) .^ 2))

    # 4. Compute the fractal dimension (approximation)
    sfd = 1 + log(L) / log(2 * (n - 1))
   
    return sfd

end

function calculate_fractal_sevcik_graph(model)
    # 1. Normalize the signal (new range to [0, 1])
    # y⭐ = vcat(map(a -> vcat(a.pos...), model.agents)...) ./ 5.0
    y⭐ = flatten_graph(model.properties.graph)

    n = length(y⭐)
    
    # 2. Derive x* and y* (y* is actually the normalized signal)
    x⭐ = LinRange(0.0, 1.0, n)

    # 3. Compute L (because we use np.diff, hence n-1 below)
    L = sum(sqrt.(diff(y⭐) .^ 2 + diff(x⭐) .^ 2))

    # 4. Compute the fractal dimension (approximation)
    sfd = 1 + log(L) / log(2 * (n - 1))

    return sfd
end

function calculate_fractal_sevcik_position(model)
    # 1. Normalize the signal (new range to [0, 1])
    y⭐ = vcat(map(a -> vcat(a.pos...), model.agents)...) ./ 5.0

    n = length(y⭐)
    
    # 2. Derive x* and y* (y* is actually the normalized signal)
    x⭐ = LinRange(0.0, 1.0, n)

    # 3. Compute L (because we use np.diff, hence n-1 below)
    L = sum(sqrt.(diff(y⭐) .^ 2 + diff(x⭐) .^ 2))

    # 4. Compute the fractal dimension (approximation)
    sfd = 1 + log(L) / log(2 * (n - 1))

    return sfd
end



function calculate_fractal_sevcik_angle(model)

    # 1. Normalize the signal (new range to [0, 1])
    y⭐ = map(a -> atan(a.vel[2], a.vel[1]) / 2π,model.agents)
    
    n = length(y⭐)
    # 2. Derive x* and y* (y* is actually the normalized signal)
    x⭐ = LinRange(0.0, 1.0, n)

    # 3. Compute L (because we use np.diff, hence n-1 below)
    L = sum(sqrt.(diff(y⭐) .^ 2 + diff(x⭐) .^ 2))

    # 4. Compute the fractal dimension (approximation)
    sfd = 1 + log(L) / log(2 * (n - 1))

    return 1.0 - (sfd  - 1.0)

end

function calculate_toroidal_coords(x,y)

    v1 = sin(x/2pi)
    v2 = cos(x/2pi)
    v3 = sin(y/2pi)
    v4 = cos(y/2pi)

    return (v1,v2,v3,v4)

end

function calculate_2D_coords(x,y,z,w)
    
    v1 = atan(x, y) / 2pi
    v2 = atan(z, w) / 2pi

    return (v1, v2)
    
end


function projectToCliffordTorus(x, y)
    R = 2
    r = 1
    theta = 2 * pi * x
    phi = 2 * pi * y
    toroidal_radius = r * cos(theta)
    phi = phi + toroidal_radius / R
    x = (R + r * cos(theta)) * cos(phi)
    y = (R + r * cos(theta)) * sin(phi)
    z = r * sin(theta)
    return (x, y, z)
end