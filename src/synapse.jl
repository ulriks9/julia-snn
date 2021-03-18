include("params.jl")
#computes the change in weight
function stdp(pre, post, p)
    delta_t = pre - post
    if delta_t < 0
        p.a_plus * exp(delta_t / p.tau_plus)
    else
        -p.a_minus * exp(delta_t / p.tau_minus)
    end
end
