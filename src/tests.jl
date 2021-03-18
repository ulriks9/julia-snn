include("dependencies.jl")
include("network.jl")
include("inputs.jl")
include("params.jl")

global input_res = length(get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")[1,:])

function get_params()
    Params(#= dt =# 1 / 20, #= tau =# 4, #= v_t =# 20, #= v_0 =# -70, #= v =# -55, #= ref =# 0.00, #= cl =# 10, #= hid =# 20, #= win =# 25, #= a_plus =# 1, #= a_minus =# 1, #= tau_plus =# 20, #= tau_minus =# 20)
end
#simulates voltage levels
function v_sim()
    layers = get_layers()
    synapses = get_synapses()
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    println("cycle method time:")
    @time l = cycle(layers, inputs, synapses, p)

    v_levels = get_parray(l[1], 1, 1, 100)

    println("plot time:")
    @time plot(v_levels)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#simulates spikes
function s_sim()
    layers = get_layers()
    synapses = get_synapses()
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    @time l = cycle(layers, inputs, synapses, p)

    s = get_parray(l[2], 3, 1, 100)

    @time plot(s)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#returns array of layers
function get_layers()
    p = get_params()
    [fill(p.v, input_res), fill(p.v, p.hid), fill(p.v, p.cl)]
end
#returns array of synapses with random floating point values in range 0:1
function get_synapses()
    p = get_params()

    layer_1 = Array{Array}(UndefInitializer(), p.hid)
    layer_2 = Array{Array}(UndefInitializer(), p.cl)

    for i = 1 : length(layer_1)
        layer_1[i] = rand(input_res)
    end
    for i = 1 : length(layer_2)
        layer_2[i] = rand(p.hid)
    end
    [layer_1, layer_2]
end
#converts array from network to something Plots.jl can work with
function get_parray(in, l, n, c)
    a = zeros(length(in))
    for i = 1 : length(in)
        a[i] = in[i][l][n]
    end
    resize!(a, c)
end
