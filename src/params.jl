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
    #summing constant
    s::Float64
    #refractory period
    ref::Float64
    #number of classes
    cl::Int8
    #number of hidden neurons
    hid::Int8
end
