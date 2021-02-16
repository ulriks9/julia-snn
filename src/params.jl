struct Params
    #size of timestep, set to 1 for the timestep size of the song
    dt::Float64
    #time constant
    tau::Int8
    #threshold voltage in mV
    v_t::Int8
    #resting voltage in mV
    v_0::Int8
    #start voltage
    v::Int8
end
