function compute(in, dt, tau, v_t, v_0, v)
    dv = -1 * (v - v_0) / tau
    dv += in
    v += dv * dt

    v >= v_t ? v_t : v
end
