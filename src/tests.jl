using Plots
using CUDA
include("network.jl")
include("inputs.jl")

global input_res = length(gen_inputs("media\\training\\ytrk.wav")[:,1])

function sim_vlevels(n)
    l = cycle(layer_1, inputs, synapses)
    l_t = l[1,n,:]
    plot(l_t)
end

function v_sim()
    layers = getLayers()
    synapses = getSynapses()
    inputs = gen_inputs("media\\training\\ytrk.wav")

    l = cycle(layers, inputs, synapses)

    v_levels = l[1]

    plot(v_levels[2,1,:])
end

function s_sim()
    layers = getLayers()
    synapses = getSynapses()
    inputs = getInputs()

    l = cycle(layers, inputs, synapses)

    spikes = l[2]

    plot(v_levels[1,1,:])
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
