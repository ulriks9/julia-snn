include(string(@__DIR__)[1:24] * "\\dependencies.jl")
include("synapse.jl")
include("neuron.jl")
include("params.jl")

#sets neurons spiked to reset
function l_reset(l, p::Params)
    for i = 1:length(l)
        if l[i] >= p.v_t
            l[i] = p.v_0
        end
    end
    l
end

function cycle(layers, inputs, synapses, p::Params)
    #simulation time, set to the amount of timesteps of the song
    t_s = length(inputs[:,1])
    #input resolution of environment, set to the amount of frequencies of the song
    input_res = length(inputs[1,:])
    #array of spike times
    spikes = zeros(input_res, length(layers[1,:]), convert(Int, t_s / p.dt))
    #monitoring of voltage levels
    v_array = zeros(input_res, length(layers[1,:]), convert(Int, t_s / p.dt))
    #refractory periods
    ref = zeros(input_res, length(layers[1,:]))
    #index
    i = 1
    #time set to 0
    t = 1
    #position
    pos = 1
    #set starting voltage
    v = p.v

    while t < t_s - p.dt
        layers = l_reset(layers, p)

        for m = 1 : length(layers[1,:])
            for n = 1 : input_res
                #checks if neuron is recovering from a spike
                if ref[n,m] > 0
                    v_array[n,m,i] = p.v_0
                    layers[n,m] = p.v_0
                    spikes[n,m,i] = 0
                    ref[n,m] -= 1
                    continue
                end
                #different step for first layer
                if m == 1
                    #computes new voltage level
                    v = layers[n,m] + compute(inputs[n,pos], layers[n,m], synapses[n,m], p)
                else
                    #sums outputs from previous layer NOT SURE HOW TO SUM
                    for o = 1 : input_res
                        #v += p.s * compute(spikes[o,m-1,i], layers[n,m], synapses[n,m], p)
                        vt = compute(spikes[o,m-1,i], layers[n,m], synapses[n,m], p)
                        #net activation from input neurons
                        layers[n,m] += vt
                    end
                    v = layers[n,m]
                end
                #if voltage is above threshold, produce spike
                if v >= p.v_t
                    spikes[n,m,i] = p.v_t
                    layers[n,m] = p.v_t
                    v_array[n,m,i] = p.v_t
                    ref[n,m] = 1
                else
                    v_array[n,m,i] = v
                    layers[n,m] = v
                end
            end
        end
        t += p.dt
        i += 1

        if t % 1 == 0
            pos += 1
        end
    end
    [v_array, spikes]
end
