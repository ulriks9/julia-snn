include("params.jl")
#computes the change in weight
function stdp(pre, post, p)
    delta_t = pre - post
    if delta_t < 0
        p.a_plus * exp(delta_t / p.tau_plus)
    else
        -p.a_minus * exp(-delta_t / p.tau_minus)
    end
end
#resets the synapses to the standard of get_synapses()
function reset_synapses()
    save_arr(get_synapses(100), "data\\synapses.jld")
end
#returns array of synapses with random floating point values from 0 to max
function get_synapses(max)
    p = get_params()

    layer_1 = Array{Array}(UndefInitializer(), p.hid)
    layer_2 = Array{Array}(UndefInitializer(), p.cl)

    for i = 1 : length(layer_1)
        layer_1[i] = rand(input_res)
        layer_1[i] = scale(layer_1[i], 0, max)
    end
    for i = 1 : length(layer_2)
        layer_2[i] = rand(p.hid)
        layer_2[i] = scale(layer_2[i], 0, max)
    end
    [layer_1, layer_2]
end
#builds array for storing traces
function get_traces()
    p = get_params()

    layer_1 = Array{Array}(UndefInitializer(), p.hid)
    layer_2 = Array{Array}(UndefInitializer(), p.cl)

    for i = 1 : length(layer_1)
        layer_1[i] = zeros(input_res)
    end
    for i = 1 : length(layer_2)
        layer_2[i] = zeros(p.hid)
    end
    [layer_1, layer_2]
end
#decays traces in array
function decay(traces, p)
    for i = 1 : length(traces), j = 1 : length(traces[i])
        traces[i][j] = map(x -> x - (p.dt * x / p.tau_g), traces[i][j])
    end
    traces
end
