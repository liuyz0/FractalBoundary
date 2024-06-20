# float64

using Distributed, MAT, ClusterManagers
#addprocs(4) # add 4 cores, different on cluster!
addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))
println("Added workers: ", nworkers())
flush(stdout)

@everywhere begin
    using LinearAlgebra, Statistics
    function lossf(x,δ,wl)
        return x.^2 .+ δ * cos.(2*π*x/wl)
    end
    function gradf(x,delta,wl)
        return 2*x .- 2*π .* delta .* sin.(2*π*x / wl) / wl
    end
    function GD(x0, lr, steps, delta, wl)
        x_traj = zeros(steps+1)
        x_traj[1] = x0
        loss_traj = - ones(steps+1)
        loss_traj[1] = lossf(x0,delta,wl)
        for i in 1:steps
            x_traj[i+1] = x0 - lr * gradf(x0,delta,wl)
            x0 = x_traj[i+1]
            loss_traj[i+1] = lossf(x0,delta,wl)
            if loss_traj[i+1] > 1e+8
                loss_traj[i+1] = -1
                break
            end
        end
        return loss_traj, x_traj
    end
end

nn = 32
s_span = LinRange(0,1.5,2^nn+1)

@everywhere function runexp(ii, s_span)
    output = -1
    s = s_span[ii]
    loss_traj, _ = GD(1.0, s, 1000, .2, .1)
    if loss_traj[end] == -1
        output = 1
    end
    println("Running to logi = ", log2(ii+1))
    flush(stdout)
    return output
end

@time output_coll = pmap(ii -> runexp(ii, s_span), 1:2^nn+1)

# save
myfilename = "./exp-1.mat"
file = matopen(myfilename, "w")
write(file, "results", output_coll)
close(file)