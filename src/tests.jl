include(string(@__DIR__)[1:24] * "\\dependencies.jl")
include("network.jl")
include("inputs.jl")
include("params.jl")

global input_res = length(get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")[1,:])

function gen_params()
    Params(#= dt =# 1 / 10, #= tau =# 4, #= v_t =# 20, #= v_0 =# -70, #= v =# -55, #= s =# 0.001, #= ref =# 0.00, #= cl =# 10)
end

function v_sim()
    layers = getLayers()
    synapses = getSynapses()
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    params = gen_params()

    l = cycle(layers, inputs, synapses, params)

    v_levels = l[1]
    v_levels = v_levels[1,1,:]
    #used for smaller plots
    v_levels = resize!(v_levels, floor(Int, length(v_levels) / 50))

    plot(v_levels)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end

function s_sim()
    layers = getLayers()
    synapses = getSynapses()
    inputs = gen_inputs("media\\training\\GTZAN\\rock\\rock.00000.wav")
    params = gen_params()

    l = cycle(layers, inputs, synapses, params)

    s = l[2]
    s = s[1,1,:]
    #used for smaller plots
    s = resize!(s, 500)

    plot(s)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end

function getLayers()
    #input layer
    layer_1 = zeros(input_res)
    fill!(layer_1, p.v)
    #hidden layer
    layer_2 = zeros(input_res)
    fill!(layer_2, p.v)
    #contains both layers
    layers = zeros(input_res, 2)
    fill!(layers, p.v)

    layers
end

function getSynapses()
    synapses = rand(input_res, 2)
    synapses[:,1] .= 1.34
    synapses[:,2] .= rand(Uniform(1000,100000))
    synapses
end

function getClasses()
    classes = zeros(p.cl)

end
