include("dependencies.jl")
include("network.jl")
include("inputs.jl")
include("params.jl")

global input_res = length(get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")[1,:])

function get_params()
    Params(#= dt =# 1 / 8, #= tau =# 10, #= v_t =# 20, #= v_0 =# -70, #= v =# -55, #= s =# 0.001, #= ref =# 0.00, #= cl =# 10, #= hid =# 20)
end

function v_sim()
    layers = get_layers()
    synapses = get_synapses()
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    l = cycle(layers, inputs, synapses, p)

    v_levels = get_parray(l[1], 1, 5, 100)

    plot(v_levels)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end

function s_sim()
    layers = get_layers()
    synapses = get_synapses()
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    l = cycle(layers, inputs, synapses, p)

    s = get_parray(l[2], 1, 5, 100)

    plot(s)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#returns array of layers
function get_layers()
    p = get_params()
    [fill(p.v, input_res), fill(p.v, p.hid), fill(p.v, p.cl)]
end
#returns array of synapses
function get_synapses()
    p = get_params()
    #first layer is set to ones, as these weights will not change
    [ones(input_res), rand(p.hid), rand(p.cl)]
end

function get_parray(in, l, n, c)
    a = zeros(length(in))
    for i = 1 : length(in)
        a[i] = in[i][l][n]
    end
    resize!(a, c)
end
