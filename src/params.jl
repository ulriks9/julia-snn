struct Params
    #size of timestep, set to 1 for the timestep size of the song
    dt::Float64
    #time constant
    tau::Float32
    #threshold voltage in mV
    v_t::Int8
    #resting voltage in mV
    v_0::Int8
    #start voltage
    v::Float64
    #refractory period
    ref::Float64
    #number of classes
    cl::Int8
    #number of hidden neurons
    hid::Int8
    #STDP window
    win::Int64
    #A plus STDP parameter
    a_plus::Float64
    #A minus STDP parameter
    a_minus::Float64
    #tau plus STDP parameter
    tau_plus::Float64
    #tau minus STDP parameter
    tau_minus::Float64
    #weight for input neurons (scales inputs)
    in_w::Float64
    #STDP initial trace
    tr::Float64
    #STDP learning rate
    l_rate::Float64
    #STDP depression rate
    d_rate::Float64
    #STDP target trace
    tar::Float64
    #STDP max weight
    w_max::Float64
    #STDP weight dependence
    w_d::Float64
    #STDP synapse decay time constant
    tau_g::Float64
    #LIF decay coefficient
    decay::Float64
end
