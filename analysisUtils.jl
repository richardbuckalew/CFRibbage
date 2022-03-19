using Dates, JSON

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




function loadsnapshot(n::Int64)
    return deserialize("snapshots/snapshot_" * string(n) * ".jls")
end


function getdeltas()

    sn = Matrix(loadsnapshot(1))
    D = []
    for ii in 2:8
        newsn = Matrix(loadsnapshot(ii))
        push!(D, norm(newsn .- sn))
        sn = newsn
    end


    return D

end










function savesnapshot()

    (ddeals, dmin, dmax, dcoverage, pdeals, pmin, pmax, pcoverage) = dealcoverage_global()
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




