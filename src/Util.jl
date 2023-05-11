function flatten_graph(g::SimpleGraph)
    return vcat(g.fadjlist...) ./ nv(g)
end