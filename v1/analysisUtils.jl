using Dates, JSON, Plots, StatsBase, StatsPlots, LaTeXStrings, ProgressMeter

function dealcoverage_global()

    N = length(keys(handID))
    ddeals = 0
    dmin = 99999
    dmax = 0
    dcovered = 0
    pdeals = 0
    pmin = 99999
    pmax = 0
    pcovered = 0

    for (H, K) in handID
        d = maximum(db.dealerplaycount[K[1]:K[2]])
        (d > dmax) && (dmax = d)
        (d < dmin) && (dmin = d)
        ddeals += d
        (d > 0) && (dcovered += 1)

        p = maximum(db.poneplaycount[K[1]:K[2]])
        (p > pmax) && (pmax = p)
        (p < pmin) && (pmin = p)
        (p > 0) && (pcovered += 1)
        pdeals += p
    end

    dcoverage = dcovered / N
    pcoverage = pcovered / N

    return (ddeals, dmin, dmax, dcoverage, pdeals, pmin, pmax, pcoverage)

end

function dealcoverage_local(db)

    N = length(keys(hRows))
    ddeals = 0
    dmin = 99999
    dmax = 0
    dcovered = 0
    pdeals = 0
    pmin = 99999
    pmax = 0
    pcovered = 0

    for (H, K) in hRows
        d = maximum(db.dealerplaycount[K[1]:K[2]])
        (d > dmax) && (dmax = d)
        (d < dmin) && (dmin = d)
        ddeals += d
        (d > 0) && (dcovered += 1)

        p = maximum(db.poneplaycount[K[1]:K[2]])
        (p > pmax) && (pmax = p)
        (p < pmin) && (pmin = p)
        (p > 0) && (pcovered += 1)
        pdeals += p
    end

    dcoverage = dcovered / N
    pcoverage = pcovered / N

    return (ddeals, dmin, dmax, dcoverage, pdeals, pmin, pmax, pcoverage)

end




function loadsnapshot(n::Int64)
    return deserialize("snapshots/snapshot_" * string(n) * ".jls")
end


function getsnapshots()

    sn = Matrix(loadsnapshot(1))
    D = Vector{Float64}[]
    P = Vector{Float64}[]
    ii = 1
    while true
        fn = "snapshot_" * string(ii) * ".jls"
        if fn in readdir("snapshots")
            x = deserialize("snapshots/" * fn)
            push!(D, x[:,1])
            push!(P, x[:,2])
        else
            break
        end
        ii += 1

    end
    return (D, P)

end





function savesnapshot(db)

    (ddeals, dmin, dmax, dcoverage, pdeals, pmin, pmax, pcoverage) = dealcoverage_local(db)
    profilesnapshot = db[:, [:dealerprofile, :poneprofile]]

    n = 1
    for filename in readdir("snapshots")
        if occursin("snapshot", filename)
            n = max(n, parse(Int64, filename[end-4])) + 1
        end
    end

    sdata = OrderedDict("nSnapshot" => n, "nDeals" => max(ddeals, pdeals), "timestamp" => now(),
                 "dCoverage" => dcoverage, "dMin" => dmin, "dMax" => dmax,
                 "pCoverage" => pcoverage, "pMin" => pmin, "pMax" => pmax)

    open("snapshots/snapdata.txt", "a") do io
        write(io, json(sdata)) + write(io, "\n")
    end

    serialize("snapshots/snapshot_" * string(n) * ".jls", profilesnapshot)
        

end


function saverecords(records)

    n = 1
    for filename in readdir("snapshots")
        if occursin("record", filename)
            n = max(n, parse(Int64, filename[8:end-4])) + 1
        end
    end
    serialize("snapshots/record_" * string(n) * ".jls", records)

end






function convergence_per_hand(snapshots)

    n = length(snapshots)
    nh = length(hRows)
    NN = zeros(Float64, nh, n-1)
    res = 101
    X = zeros(Float64, res, n-1)

    @showprogress 1 for (nhand, (h, rows)) in enumerate(pairs(hRows))

        V = [S[rows[1]:rows[2]] for S in snapshots]
        N = norm.([V[ii] - V[end] for ii in 1:(n-1)])
        NN[nhand, :] = N

    end

    maxnorm = maximum(NN)

    for ii in 1:(n-1)
        
        for nhand in 1:nh

            jj = floor(Int64, (res-1) * NN[nhand, ii] / maxnorm) + 1
            X[jj,ii] += 1

        end

    end

    heatmap(1:16, range(0, 1, 101), log.(X))
    plot!(xlabel = "Snapshot #", ylabel=L"|\Delta|", title = "Log norm frequency per snapshot")

    # X ./= nh

    # heatmap(X, zscale=:log10)


    # mm = minimum(NN, dims=2)
    # M = mean(NN, dims=2)
    # S = std(NN, dims=2)
    # MM = maximum(NN, dims = 2)

    # display(M)
    # display(S)

    # plot([mm M-S M M+S MM], label=["min" "-1 std" "mean" "+1 std" "max"])
    # plot!(title="norm(δ) averaged across all hands")

    return X

end


function convergence(snapshots)

    n_snapshots = length(snapshots)
    n_hands = length(hRows)

    deltas = [ ]
    nzeros = zeros(Int64, n_snapshots-1)


    for n_snap in 1:(n_snapshots-1)

        X = Float64[]
        sizehint!(X, 1_000_000)

        for hand in allhands

            rows = hRows[hand]

            v_old = snapshots[n_snap][rows[1]:rows[2]]
            v_new = snapshots[n_snap+1][rows[1]:rows[2]]

            d = norm(v_old .- v_new, 1)


            d ≈ 0 ? nzeros[n_snap] += 1 : push!(X, d)

        end

        push!(deltas, X)

    end

    return (deltas, nzeros)

end

function showconvergence(snapshots)

    (D, P) = snapshots

    l = @layout [a; b]

    (deltas, nzeros) = convergence(D)

    nhands = length(allhands)

    p1 = boxplot(deltas, legend=false)
    p1 = plot!(nzeros ./ nhands, lw = 2, lc = :black)

    (deltas, nzeros) = convergence(P)

    p2 = boxplot(deltas, legend=false)
    p2 = plot!(nzeros ./ nhands, lw = 2, lc = :black)

    plot(p1, p2, layout = l)

end







function analyzerecords(records::Vector{PlayRecord})
    
    Ydealer = summarystats((r.dealerdelta for r in records))
    Ypone = summarystats((r.ponedelta for r in records))

    return (Ydealer, Ypone)
    
end




