# additive noise frac dim vs dim

using Distributed, MAT, ClusterManagers
#addprocs(4) # add 4 cores, different on cluster!
addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))
println("Added workers: ", nworkers())
flush(stdout)

@everywhere begin
    using LinearAlgebra, Statistics
    function lossf(x,δ,wl)
        return sum(x.^2 .+ δ * cos.(2*π*x/wl))
    end
    function gradf(x,delta,wl)
        return 2*x .- 2*π .* delta .* sin.(2*π*x / wl) / wl
    end
    function GD(d, x0, lr, steps, delta, wl)
        x_traj = zeros(steps+1)
        x_traj[1] = x0
        loss_traj = - ones(steps+1)
        loss_traj[1] = lossf(x0,delta,wl)
        for i in 1:steps
            x_traj[i+1] = x0 - lr * gradf(x0,delta,wl)
            x0 = x_traj[i+1]
            loss_traj[i+1] = lossf(x0,delta,wl)
            if loss_traj[i+1] > 1e+8 * d
                loss_traj[i+1] = -1
                break
            end
        end
        return loss_traj, x_traj
    end
end

nn = 20
s_span = LinRange(0,1.5,2^nn+1)
num_dim = 100

@everywhere function runexp(ii, s_span, num_dim)
    output = - ones(num_dim)
    s = s_span[ii]

    for i in 1:num_dim
        loss_traj, _ = GD(i,ones(i)/i, s, 1000, .2, .1)
        if loss_traj[end] == -1
            output[i] = 1
        end
    end
    println("Running to logi = ", log2(ii+1))
    flush(stdout)
    return output
end

@time output_coll = pmap(ii -> runexp(ii, s_span, num_dim), 1:2^nn+1)
outputs = zeros(2^nn+1,num_dim)

for ii in 1:2^nn+1
    outputs[ii,:] = output_coll[ii]
end

# save
myfilename = "./exp-5.mat"
file = matopen(myfilename, "w")
write(file, "results", outputs)
close(file)