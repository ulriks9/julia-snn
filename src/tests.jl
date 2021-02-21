include(string(@__DIR__)[1:24] * "\\dependencies.jl")
include("network.jl")
include("inputs.jl")

global input_res = length(gen_inputs("media\\training\\ytrk.wav")[:,1])

function gen_params()
    Params(#= dt =# 1 / 2, #= tau =# 1, #= v_t =# 30, #= v_0 =# -70, #= v =# -70, #= s =# 0.1)
end

function v_sim()
    layers = getLayers()
    synapses = getSynapses()
    inputs = gen_inputs("media\\training\\GTZAN\\rock\\rock.00000.wav")
    #scale up inputs (should change to normalization)
    inputs = scale(inputs, 0, 500)
    params = gen_params()

    l = cycle(layers, inputs, synapses, params)

    v_levels = l[1]
    v_levels = v_levels[1,1,:]
    v_levels = resize!(v_levels, 100)

    plot(v_levels)
end

function s_sim()
    layers = getLayers()
    synapses = getSynapses()
    inputs = gen_inputs("media\\training\\ytrk.wav")
    params = gen_params()

    l = cycle(layers, inputs, synapses, params)

    spikes = l[2]

    plot(spikes[1,1,:])
end

function getLayers()
    v = -55
    #input layer
    layer_1 = zeros(input_res)
    fill!(layer_1, v)
    #hidden layer
    layer_2 = zeros(input_res)
    fill!(layer_2, v)
    #contains both layers
    layers = zeros(input_res, 2)
    fill!(layers, v)

    layers
end

function getSynapses()
    #array storing synapses for each neuron
    synapses = rand(input_res, 2)
    #returns scaled values for weights
    synapses = map(x -> x * 100, synapses)

    synapses
end

function getInputs()
    #array of input spikes from synapses
    inputs = zeros(input_res, 2, convert(Int, t_s / dt))
    #assigns random number to each input
    inputs = map(x -> x + rand() * 1, inputs)
    #removes values less than 0 (not necessary yet)
    inputs = map(x -> x < 0 ? x = 0 : x = x, inputs)
    #randomly adds high inputs based on a probability
    inputs = map(x -> rand() > 0.99 ? x = x + 100 : x = x, inputs)

    inputs
end
#scales array between specified range
function scale(arr, a, b)
    max = maximum(arr)
    min = minimum(arr)

    for i = 1:length(arr)
        arr[i] = ((b - a) * (arr[i] - min) / (max - min)) + a
    end
    arr
end
