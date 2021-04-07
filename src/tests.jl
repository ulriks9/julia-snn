include("dependencies.jl")
include("network.jl")
include("inputs.jl")
include("params.jl")
include("storage.jl")
include("synapse.jl")

global input_res = length(get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")[1,:])

#returns an object of type Params which stores the hyperparameters for the network
function get_params()
    Params(#= dt =# 1 / 5, #= tau =# 0.15, #= v_t =# 30, #= v_0 =# -70, #= v =# -55,
        #= ref =# 0.00, #= cl =# 10, #= hid =# 200, #= in_w =# 0.5, #= tr =# 0.25, #= l_rate =# 1e-7,
        #= d_rate =# 1e-7, #= tar =# 0, #= w_max =# 10, #= w_d =# 2, #= tau_g =# 5, #= decay =# 1e-2)
end

#Runs n amount of training epochs with random samples
function train(n::Int64)
    samples = String[]
    itr = walkdir("media\\training\\GTZAN")
    f = first(itr)
    #creates array of paths of all samples
    for i in itr
        for j = 1 : length(i[3])
            push!(samples, i[1] * "\\" * i[3][j])
        end
    end

    for i = 1 : n
        println("")
        println("------------------------------------------------------------------------")
        println("")
        println("Epoch " * string(i) * ":")
        println("")

        r = rand(1:length(samples))
        sample = samples[r]
        deleteat!(samples, r)

        println("Sample processed: " * sample)
        println("")

        @time first = run_cycle(sample, true)
        println("")

        println("First neuron to spike: " * string(first))
    end
end
#trains the network on every sample of a genre
function train_genre(genre::String)
    itr = walkdir("media\\training\\GTZAN")
    genres = first(itr)[2]
    samples = Array{Array}(UndefInitializer(), length(genres))

    for i = 1 : length(genres)
        samples[i] = String[]
    end

    count = 1
    for i in itr
        for j = 1 : length(i[3])
            push!(samples[count], i[1] * "\\" * i[3][j])
        end
        count += 1
    end

    training_samples = samples[findfirst(isequal(genre), genres)]

    for i = 1 : length(training_samples)
        println("")
        println("------------------------------------------------------------------------")
        println("")
        println("Epoch " * string(i) * ":")
        println("")

        r = rand(1:length(training_samples))
        sample = training_samples[r]
        deleteat!(training_samples, r)

        println("Sample processed: " * sample)
        println("")

        @time first = run_cycle(sample, true)
        println("")

        println("First neuron to spike: " * string(first))
    end
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
#used for finding an invalid WAV files in the dataset
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
#simulates membrane potential of the first neuron of specified layer
function v_sim(layer::Int64)
    layers = get_layers()
    synapses = load_arr("data\\synapses.jld")
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    println("cycle method time:")
    @time l = cycle(layers, inputs, synapses, p, false)

    v_levels = get_parray(l[1], layer, 1, 500)

    println("plot time:")
    @time plot(v_levels)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#simulates spikes of the first neuron of the first layer
function s_sim()
    layers = get_layers()
    synapses = load_arr("data\\synapses.jld")
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    @time l = cycle(layers, inputs, synapses, p)

    s = get_parray(l[2], 1, 1, 500)

    @time plot(s)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#returns array of layers
function get_layers()
    p = get_params()
    [fill(p.v, input_res), fill(p.v, p.hid), fill(p.v, p.cl)]
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
function run_cycle(path::String, stdp)
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

    first = 0
    for i = 1 : length(s), j = 1 : length(s[i][length(s[i])])
        if s[i][length(s[i])][j] > 0
            first = j
            break
        end
    end
    [spikes,first]
end
