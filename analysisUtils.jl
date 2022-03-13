using Dates, JSON

function dealcoverage()

    N = length(keys(handID))
    ddeals = 0
    dcovered = 0
    pdeals = 0
    pcovered = 0

    for (H, K) in handID
        d = maximum(db.dealerplaycount[K[1]:K[2]])
        ddeals += d
        (d > 0) && (dcovered += 1)
        p = maximum(db.poneplaycount[K[1]:K[2]])
        (p > 0) && (pcovered += 1)
        pdeals += p
    end

    dcoverage = dcovered / N
    pcoverage = pcovered / N

    return (ddeals, dcoverage, pdeals, pcoverage)

end


function profilesnapshot()

    return db[:, [:dealerprofile, :poneprofile]]

end


function progressreport()

    (ddeals, dcoverage, pdeals, pcoverage) = dealcoverage()
    snapshot = profilesnapshot()

    n = 1
    for filename in readdir("snapshots")
        if occursin("snapshot", filename)
            n = max(n, parse(Int64, filename[end-3]))
        end
    end

    sdata = Dict("nSnapshot" => n, "nDeals" => max(ddeals, pdeals), "dCoverage" => dcoverage, "pCoverage" => pcoverage, "timestamp" => now())

    open("snapshots/snapdata.txt", "w") do io
        write(io, json(sdata))
    end

    serialize("snapshots/snapshot_" * string(n) * ".jls", snapshot)
        

end




