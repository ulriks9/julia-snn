include(string(@__DIR__)[1:24] * "\\dependencies.jl")
include("synapse.jl")
include("neuron.jl")
include("params.jl")

function cycle(layers, inputs, synapses, p::Params)
    #simulation time, set to the amount of timesteps of the song
    t_s = length(inputs[1,:])
    #input resolution of environment, set to the amount of frequencies of the song
    input_res = length(inputs[:,1])
    #array of spike times
    spikes = zeros(input_res, length(layers[1,:]), convert(Int, t_s / p.dt))
    #monitoring of voltage levels
    v_array = zeros(input_res, length(layers[1,:]), convert(Int, t_s / p.dt))
    #index
    i = 1
    #time set to 0
    t = 1
    #position
    pos = 1
    #set starting voltage
    v = p.v

    while t < t_s - p.dt
        for m = 1 : length(layers[1,:])
            for n = 1 : input_res
                #if neuron spiked previously, set its voltage to resting
                if layers[n,m] > p.v_t
                    layers[n,m] = p.v_0
                end
                #different step for first layer
                if m == 1
                    #computes next voltage level
                    v = compute(inputs[n,pos], p.dt, p.tau, p.v_t, p.v_0, layers[n,m])
                    #computes weighted voltage
                    v = compute(v, synapses[n,m])
                else
                    #sums outputs from previous layer NOT SURE HOW TO SUM
                    for o = 1 : input_res
                        #computes next voltage level
                        vt = compute(layers[o,m-1], p.dt, p.tau, p.v_t, p.v_0, layers[n,m])
                        #computes weighted voltage
                        v += compute(vt, synapses[o,m])
                    end
                end

                #Current injection, only used for simulating neurons individually
                if t > 2 && t < 4
                    #v += 3.5
                end

                #used mostly for logging
                v_array[n,m,i] = v

                #if voltage is above threshold, produce spike
                if v_array[n,m,i] >= p.v_t
                    spikes[n,m,i] = p.v_t
                    #not sure if this should be v_0 or compute(v_0, synapses[n,m])
                    layers[n,m] = compute(p.v_0, synapses[n,m])
                end

                #used for temporarily storing postsynaptic voltage levels
                layers[n,m] = v
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
