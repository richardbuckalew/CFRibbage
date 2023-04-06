module CFRibbage


include("base.jl")
include("analytics.jl")
include("training.jl")
# include("web.jl")

using BenchmarkTools, ProfileView
using JSON, Random





function dealHands(db::DB, skiplist_dealer::Vector{Int}, skiplist_pone::Vector{Int})


    while true

        shuffle!(db.deck)    

        hand1 = db.deck[1:6]
        hand2 = db.deck[7:12]
        turncard = db.deck[13] 

        (h1, sp1) = canonicalize(hand1)
        (h2, sp2) = canonicalize(hand2)

        hid1 = db.hID[h1]
        hid2 = db.hID[h2]

        if !(hid1 in skiplist_dealer) || !(hid2 in skiplist_pone)
            return (hand1, hand2, turncard, h1, h2, sp1, sp2)
        end
    end

end


"Do one core CFR loop."
function CFR!(playbuffer1::Vector{Int64}, playbuffer2::Vector{Int64},
             showbuffer1::Vector{Int64}, showbuffer2::Vector{Int64},
             profilebuffer1::Vector{Float64}, profilebuffer2::Vector{Float64},
             playprobbuffer1::Vector{Float64}, playprobbuffer2::Vector{Float64},
             rows1, rows2, h1::hType, h2::hType, sp1, sp2, di1::Int64, di2::Int64, turncard,
             known1::HType, known2::HType, IMs1::Vector{InformationModel_Base}, IMs2::Vector{InformationModel_Base}, db::DB)


    # println("CFR")

    getPlayScores!(playbuffer1, playbuffer2, rows1, rows2, di1, di2, known1, known2, turncard.rank, IMs1, IMs2, db)
    getShowScores!(showbuffer1, showbuffer2, rows1, rows2, h1, sp1, h2, sp2, di1, di2, turncard)
    
    # println(showbuffer1)
    # println(showbuffer2)
    # println(showbuffer1[di1], "  ?=  ", showbuffer2[di2])
    # println("\n")


    n1 = nrow(rows1)
    n2 = nrow(rows2)

    profilebuffer1[1:n1] = rows1[:, :profile_dealer]
    profilebuffer2[1:n2] = rows2[:, :profile_pone]


    rows1[1, :dealt_dealer] += 1
    rows2[1, :dealt_pone] += 1

    # update regrets
    rows1[:, :regret_dealer] .= ((rows1[1, :dealt_dealer] - 1) .* rows1[:, :regret_dealer] .+ playbuffer1[1:n1] .+ showbuffer1[1:n1] .- playbuffer1[di1] .- showbuffer1[di1]) ./ rows1[1, :dealt_dealer]
    rows2[:, :regret_pone] .= ((rows2[1, :dealt_pone] - 1) .* rows2[:, :regret_pone] .+ playbuffer2[di2] .+ showbuffer2[di2] .- playbuffer2[1:n2] .- showbuffer2[1:n2]) ./ rows2[1, :dealt_pone]

    # update profiles
    if any((x -> x > 0), rows1[:, :regret_dealer])
        rows1[:, :profile_dealer] .= map((x -> max(x, 0.0)), rows1[:, :regret_dealer]) ./ sum((x -> max(x, 0.0)), rows1[:, :regret_dealer])
    else
        rows1[:, :profile_dealer] .= 1.0 ./ n1
    end
    if any((x -> x > 0), rows2[:, :regret_pone])
        rows2[:, :profile_pone] .= map((x -> max(x, 0.0)), rows2[:, :regret_pone]) ./ sum((x -> max(x, 0.0)), rows2[:, :regret_pone])
    else
        rows2[:, :profile_pone] .= 1.0 ./ n2
    end

    # update overall play probabilities
    playprobbuffer1[1:n1] = rows1[:, :p_play_dealer]
    rows1[:, :p_play_dealer] = rows1[:, :p_deal] .* rows1[:, :profile_dealer]
    playprobbuffer2[1:n2] = rows2[:, :p_play_pone]
    rows2[:, :p_play_pone] = rows2[:, :p_deal] .* rows2[:, :profile_pone]

    for (ix, ph) in enumerate(rows1[:, :playhand])
        db.Hprobs_dealer[counter(ph)] += rows1[ix, :p_play_dealer] - playprobbuffer1[ix]
    end
    for (ix, ph) in enumerate(rows2[:, :playhand])
        db.Hprobs_pone[counter(ph)] += rows2[ix, :p_play_pone] - playprobbuffer2[ix]
    end
    
end



"Train a batch of size n"
function doBatch(n::Int64, db::DB)#, force_deal_threshold = 1.0)

    # force_hand = nothing

    # objects for creating tData at the end
    dealt_dealer = Vector{Int64}(undef, n)
    dealt_pone = Vector{Int64}(undef, n)
    discards_dealer = Vector{Int64}(undef, n)
    discards_pone = Vector{Int64}(undef, n)
    scores = Vector{Int64}(undef, n)
    deltas_dealer = Vector{Float64}(undef, n)
    deltas_pone = Vector{Float64}(undef, n)
    # mean_delta = 0.0


    # reusable objects for CFR
    pb1 = fill(0, 20)
    pb2 = fill(0, 20)
    sb1 = fill(0, 20)
    sb2 = fill(0, 20)
    pfb1 = fill(0.0, 20)
    pfb2 = fill(0.0, 20)
    ppb1 = fill(0.0, 20)
    ppb2 = fill(0.0, 20)

    IMs1 = [InformationModel_Base(db) for ix in 1:15]
    IMs2 = [InformationModel_Base(db) for ix in 1:15]

    skiplist_dealer = deserialize("data/skiplist_dealer.jls")
    skiplist_pone = deserialize("data/skiplist_pone.jls")

    # train!
    t0 = Base.Libc.time()
    for nhand in 1:n

        # compile hand info
        (hand1, hand2, turncard, h1, h2, sp1, sp2) = dealHands(db, skiplist_dealer, skiplist_pone)

        # println("Dealt ", hand1, " and ", hand2)

        rows1 = @view db.df[db.hRows[h1], :]
        rows2 = @view db.df[db.hRows[h2], :]
    
        di1 = getDiscard(rows1, 1)
        di2 = getDiscard(rows2, 2)

        # println("  Discards: ", di1, " ", di2)
    
        D1 = HType()
        for suit in rows1[di1,:].discard
            for r in suit
                D1[r] += 1
            end
        end
        H1 = counter(rows1[di1,:].playhand)
    
        D2 = HType()
        for suit in rows2[di2,:].discard
            for r in suit
                D2[r] += 1
            end
        end
        H2 = counter(rows2[di2,:].playhand)
    
        known1 = merge(H1, D1)
        known2 = merge(H2, D2)
        known1[turncard.rank] += 1
        known2[turncard.rank] += 1


        fill!(pb1, 0.0)
        fill!(pb2, 0.0)
        fill!(sb1, 0.0)
        fill!(sb2, 0.0)

        # do the CFR
        CFR!(pb1, pb2, sb1, sb2, pfb1, pfb2, ppb1, ppb2, rows1, rows2, h1, h2, sp1, sp2, di1, di2, turncard, known1, known2, IMs1, IMs2, db)

        # update tData
        dealt_dealer[nhand] = db.hID[h1]
        dealt_pone[nhand] = db.hID[h2]
        discards_dealer[nhand] = di1
        discards_pone[nhand] = di2
        scores[nhand] = pb1[di1] + sb1[di1]
        deltas_dealer[nhand] = maximum(abs.(pfb1[1:nrow(rows1)] - rows1[:, :profile_dealer]))
        deltas_pone[nhand] = maximum(abs.(pfb2[1:nrow(rows2)] - rows2[:, :profile_pone]))


        # if deltas_dealer[nhand] > force_deal_threshold
        #     force_hand = hand1
        #     # print('.')
        # elseif deltas_pone[nhand] > force_deal_threshold
        #     force_hand = hand2
        #     # print('.')
        # else
        #     force_hand = nothing
        # end


    end
    dt = Base.Libc.time() - t0

    return (TrainingData(n, dt, dealt_dealer, dealt_pone, discards_dealer, discards_pone, scores, deltas_dealer, deltas_pone), summarize(db))
end


function saveDB!(db::DB)
    # println("Serializing M...")
    # @time serialize("data/Mpacked.jls", db.M)
    tempM = copy(db.M)
    fill!(db.M, nothing)
    println("Serializing db...")
    @time serialize("data/db.jls", db)
    Folds.map!(x->x, db.M, tempM)
end

function loadDB()
    println("Deserializing db...")
    @time db = deserialize("data/db.jls")
    println("Deserializing M...")
    @time Folds.map!(x->x, db.M, deserialize("data/Mpacked.jls"))
    return db
end



function init_environment()
    # deck = [Card(r,s) for r in 7:11 for s in 1:4]
    deck = standardDeck
    db = initDB(deck)
    buildM!(db.M, db.allH, db.HID)

    saveDB!(db)
end





function train(db, nBatches = 100, batchSize = 100000)

    println(" ")
    println(" ")
    println("Training begins.")

    # force_deal_threshold = 1.0

    nbatches_present = 0
    for fn in readdir("data/")
        if fn[1:5] == "tData"
            head = splitext(fn)[1]
            n = parse(Int64, split(head, '_')[2])
        else
            continue
        end

        # (fn == "db.jls") && continue
        # (fn == "Mpacked.jls") && continue
        # n = parse(Int64, split(splitext(fn)[1], "_")[2])
        (n > nbatches_present) && (nbatches_present = n)
    end

    for ix in 1:nBatches
        println("Batch ", ix, " of ", nBatches)

        (tData, sStats) = doBatch(batchSize, db)#, force_deal_threshold)

        # μ = mean(vcat(tData.deltas_dealer, tData.deltas_pone))
        # σ = std(vcat(tData.deltas_dealer, tData.deltas_pone))

        println("Dealt ", tData.n_dealt, " hands in ", tData.dt, " s.")
        # println("  Average Δ: ", μ)
        # println("  Std Δ: ", σ)
        println("  Coverage: ", (round(sStats.coverage_dealer; digits = 4), round(sStats.coverage_pone; digits = 4)))
        println("\n")

        open("data/tData_" * string(nbatches_present + ix) * ".json", "w") do f
            JSON.print(f, tData)
        end
        open("data/sStats_" * string(nbatches_present + ix) * ".json", "w") do f
            JSON.print(f, sStats)
        end

        # force_deal_threshold = μ + σ
    end

    saveDB!(db)
end


# db = loadDB()
# train(db, 100, 100000)


# db = deserialize("data/db.jls")
# buildM!(db.M, db.allH, db.HID)
# serialize("data/Mpacked.jls", db.M)

end # module CFRibbage
