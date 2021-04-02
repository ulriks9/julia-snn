include("dependencies.jl")
include("network.jl")
include("inputs.jl")
include("params.jl")
include("storage.jl")
include("synapse.jl")

global input_res = length(get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")[1,:])

function get_params()
    Params(#= dt =# 1 / 5, #= tau =# 4, #= v_t =# 30, #= v_0 =# -70, #= v =# -55,
        #= ref =# 0.00, #= cl =# 10, #= hid =# 20, #= win =# 25, #= a_plus =# 0.05, #= a_minus =# 0.01,
        #= tau_plus =# 20, #= tau_minus =# 20, #= in_w =# 1)
end
#runs training on the network for specified amount of samples
function run_training(n::Int64)
    classes = readdir("media\\training\\GTZAN")
    processed = String[]
    spikes = Int[]

    for i = 1 : n
        println("Epoch " * string(i) * ":")
        class = classes[rand(1:length(classes))]
        samples = readdir("media\\training\\GTZAN\\" * class)
        sample = samples[rand(1:length(samples))]
        path = "media\\training\\GTZAN\\" * class * "\\" * sample

        if !(sample in processed)
            @time push!(spikes, argmax(run_cycle(path, true)))
            push!(processed, sample)
        #if a sample is skipped, reduce i to keep number of iterations the same
        else
            i -= 1
        end
    end
    hcat(processed, spikes)
end
#finds which class was most associated with an output neuron
function assign_classes(arr)
    s = String[]
    classes = readdir("media\\training\\GTZAN")
    counts = zeros(length(classes))
    tracker = hcat(classes, counts)
    neurons = Array{Int64}(1:length(classes))
    genres = Array{String}(UndefInitializer(), length(classes))
    out = hcat(neurons, genres)

    for i in 1 : length(classes)
        for j = 1 : length(arr[:,1])
            if arr[j,2] == i
                push!(s, arr[j,1])
            end
        end
        for j = 1 : length(s)
            for k = 1 : length(tracker[:,1])
                if occursin(tracker[k,1], s[j])
                    tracker[k,2] += 1
                end
            end
        end
        out[i,2] = tracker[argmax(tracker[:,2]),1]
    end
    out
end
#used for finding an invalid WAV file in the dataset
function find_faulty()
    classes = readdir("media\\training\\GTZAN")
    for i = 1 : length(classes)
        samples = readdir("media\\training\\GTZAN\\" * classes[i])
        for j = 1 : length(samples)
            println(samples[j] * ":")
            w = wavread("media\\training\\GTZAN\\" * classes[i] * "\\" * samples[j])
            println("Working")
        end
    end
end
#simulates voltage levels
function v_sim()
    layers = get_layers()
    synapses = load_arr("data\\synapses.jld")
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    println("cycle method time:")
    @time l = cycle(layers, inputs, synapses, p, false)

    v_levels = get_parray(l[1], 1, 5, 500)

    println("plot time:")
    @time plot(v_levels)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#simulates spikes
function s_sim()
    layers = get_layers()
    synapses = load_arr("data\\synapses.jld")
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    @time l = cycle(layers, inputs, synapses, p)

    s = get_parray(l[2], 1, 5, 500)

    @time plot(s)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#returns array of layers
function get_layers()
    p = get_params()
    [fill(p.v, input_res), fill(p.v, p.hid), fill(p.v, p.cl)]
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
#converts array from network to something Plots.jl can work with
function get_parray(in, l, n, c)
    a = zeros(length(in))
    for i = 1 : length(in)
        a[i] = in[i][l][n]
    end
    resize!(a, c)
end
#runs one training cycle with specified input, returns # of times output neurons spiked
function run_cycle(path::AbstractString, stdp)
    p = get_params()
    layers = get_layers()
    inputs = get_mfcc(path)
    synapses = load_arr("data\\synapses.jld")

    if stdp
        l = cycle(layers, inputs, synapses, p, true)
    else
        l = cycle(layers, inputs, synapses, p, false)
    end

    s = l[2]

    save_arr(l[3], "data\\synapses.jld")

    spikes = zeros(p.cl)

    for i = 1 : length(s)
        for j = 1 : length(s[i][3])
            if s[i][3][j] > 0
                spikes[j] += 1
            end
        end
    end
    spikes
end
