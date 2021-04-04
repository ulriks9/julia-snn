include("params.jl")
#computes the next value for v at each timestep
function compute(in, v, w, p::Params)
    dv = (-p.decay * (v - p.v_0) + w * in) / p.tau
    dv * p.dt
end
