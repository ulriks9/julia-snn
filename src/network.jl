include("dependencies.jl")
include("synapse.jl")
include("neuron.jl")
include("params.jl")
include("storage.jl")

#sets neurons spiked to reset
function l_reset(l, p::Params)
    for i = 1 : length(l)
        for j = length(l[i])
            if l[i][j] >= p.v_t
                l[i][j] = p.v_0
            end
        end
    end
    l
end

function cycle(layers, inputs, synapses, p::Params, stdp_b)
    #useful for setting parameters
    stdp_counter = 0
    #simulation time, set to the amount of timesteps of the song
    t_s = length(inputs[:,1])
    #input resolution of environment, set to the amount of frequencies of the song
    input_res = length(inputs[1,:])
    #array of spike times
    spikes = Array{Array}(UndefInitializer(), convert(Int, t_s / p.dt))
    stdp_spikes = Array{Array}(UndefInitializer(), convert(Int, t_s / p.dt))
    #monitoring of voltage levels
    v_array = Array{Array}(UndefInitializer(), convert(Int, t_s / p.dt))
    #initializes arrays with correct structure
    for x = 1 : length(spikes)
        spikes[x] = [zeros(input_res), zeros(p.hid), zeros(p.cl)]
        v_array[x] = [zeros(input_res), zeros(p.hid), zeros(p.cl)]
    end
    #refractory periods
    ref = [zeros(input_res), zeros(p.hid), zeros(p.cl)]
    #STDP traces
    pre_traces = get_traces()
    post_traces = get_traces()
    #index
    i = 1
    #time set to 0
    t = 1
    #position
    pos = 1
    #set starting voltage
    v = p.v
    #main iterator, goes through the inputs one timestep at a time
    while t < t_s - p.dt
        layers = l_reset(layers, p)
        #iterates through layers
        for m = 1 : length(layers)
            #iterates through each neuron in layer
            for n = 1 : length(layers[m])
                #checks if neuron is recovering from a spike
                if ref[m][n] > 0
                    v_array[i][m][n] = p.v_0
                    layers[m][n] = p.v_0
                    spikes[i][m][n] = 0
                    ref[m][n] -= 1
                    continue
                end
                #different step for first layer
                if m == 1
                    #computes new voltage level
                    #instead of passing a synapse as parameter, it uses p.in_w for the input layer
                    v = layers[m][n] + compute(inputs[pos,n], layers[m][n], p.in_w, p)
                else
                    #sums outputs from previous layer
                    for o = 1 : length(layers[m-1])
                        vt = compute(spikes[i][m-1][o], layers[m][n], synapses[m-1][n][o], p)
                        #net activation from input neurons
                        layers[m][n] += vt
                    end
                    v = layers[m][n]
                end
                #if voltage is above threshold, produce spike
                if v >= p.v_t
                    spikes[i][m][n] = p.v_t
                    layers[m][n] = p.v_t
                    v_array[i][m][n] = p.v_t
                    ref[m][n] = 1
                    #performs STDP if toggled
                    if stdp_b
                        if m == 1
                            #first layer neurons can only leave presynaptic traces
                            for a = 1 : length(pre_traces[m])
                                pre_traces[m][a][n] += p.tr
                            end
                            #performs weight depression
                            for a = 1 : length(post_traces[m][n])
                                #updates weight based on strength of trace left by the postsynaptic neuron
                                synapses[m][n][a] += -(p.d_rate * post_traces[m][n][a] * synapses[m][n][a] ^ p.w_d)
                                stdp_counter += 1
                            end
                        elseif m == 2
                            #leaves pre and postsynaptic traces
                            for a = 1 : length(pre_traces[m])
                                pre_traces[m][a][n] += p.tr
                            end
                            for a = 1 : length(synapses[m-1][n])
                                post_traces[m-1][n][a]
                            end
                            #performs weight potentiation
                            for a = 1 : length(pre_traces[m-1][n])
                                #updates weight based on strength of trace left by the presynaptic neuron
                                synapses[m-1][n][a] += p.l_rate * (pre_traces[m-1][n][a] - p.tar) * (p.w_max - synapses[m-1][n][a]) ^ p.w_d
                                stdp_counter += 1
                            end
                            #performs weight depression
                            for a = 1 : length(post_traces[m])
                                #updates weight based on strength of trace left by the postsynaptic neuron
                                synapses[m][a][n] += -(p.d_rate * post_traces[m][a][n] * synapses[m][a][n] ^ p.w_d)
                                stdp_counter += 1
                            end
                        else
                            #leaves postsynaptic traces
                            for a = 1 : length(post_traces[m-1][n])
                                post_traces[m-1][n][a] += p.tr
                            end
                            #performs weight potentiation
                            for a = 1 : length(pre_traces[m-1][n])
                                #updates weight based on strength of trace left by the presynaptic neuron
                                synapses[m-1][n][a] += p.l_rate * (pre_traces[m-1][n][a] - p.tar) * (p.w_max - synapses[m-1][n][a]) ^ p.w_d
                                stdp_counter += 1
                            end
                        end
                    end
                else
                    v_array[i][m][n] = v
                    layers[m][n] = v
                end
            end
        end
        t += p.dt
        i += 1
        #updates position at every whole number step
        if t % 1 == 0
            pos += 1
        end
        #decays traces at each timestep
        pre_traces = decay(pre_traces, p)
        post_traces = decay(post_traces, p)
    end
    save_arr(pre_traces, "data\\traces.jld")
    println("STDP Performed " * string(stdp_counter) * " times")
    [v_array, spikes, synapses]
end
