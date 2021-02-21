include("params.jl")
#computes the next value for v at a timestep with size dt
function compute(in, dt, tau, v_t, v_0, v)
    dv = -1 * (v - v_0) / tau
    dv += in
    v += dv * dt

    v >= v_t ? v_t : v
end
#computes the next value for v at each timestep
function compute(in, v, w, p::Params)
    dv = (-(v - p.v_0) + w * in) / p.tau
    dv * p.dt
end
