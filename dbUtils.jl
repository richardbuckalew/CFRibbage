

(@isdefined CC) || (const CC = collect.([combinations(cardranks, ii) for ii in 0:6]))
function newSuits(H)

    nSuit = length(H) + 1
    (nSuit > 1) ? (cardsleft = 6 - sum(length.(H))) : (cardsleft = 6)
    (cardsleft == 0) && (return [])

    NS = []

    if nSuit == 1
        m = 2
        M = 6
    elseif 1 < nSuit < 4
        m = Int(ceil(cardsleft / (4 - nSuit + 1)))
        M = cardsleft
    else
        m = cardsleft
        M = cardsleft
    end

    for suitSize in m:M
        append!(NS, CC[suitSize+1])
    end

    return NS

end

function addSuit(HH)


    newHH = Set([])

    for H in HH
        NS = newSuits(H)
        (length(NS) == 0) && push!(newHH, copy(H))
        for ns in NS
            G = copy(H)
            push!(G, ns)
            sort!(sort!(G), by = length)
            push!(newHH, G)
        end
    end

    return newHH

end

function getDiscards(H) ## Discard format: A tuple of tuples, one for each suit in the canonical form of the hand. 
                        ## Respects canonical suit order on a per-hand basis

    D = []

    for (i1, i2) in combinations(1:4,2)
        (isempty(H[i1]) || isempty(H[i2])) && continue;

        for (c1, c2) in product(H[i1], H[i2])

            d = [[] ,[], [], []];
            d[i1] = [c1]
            d[i2] = [c2]

            # push!(D, d);
            push!(D, Tuple(Tuple.(d)))

        end
    end

    for s in unique(H)
        isempty(s) && continue;

        ii = findfirst(Ref(s) .== H)
        (length(s) < 2) && continue
        for (c1, c2) in combinations(s, 2)
            d = [[], [], [], []]
            d[ii] = [c1, c2]

            # push!(D, d);
            push!(D, Tuple(Tuple.(d)))
        end
    end

    return D

end

function getPlayHand(h, d)
    return  Tuple(sort(vcat([Array(setdiff(h[ii], d[ii])) for ii in 1:4]...)))
end

function dealAllHands()

    # dealCounts = Dict{Any, Int64}();
    dealCounts = counter(Vector{Vector{Int64}})
    deck = standardDeck


    for comb in combinations(deck, 6)

        (H, sp) = canonicalize(comb)

        inc!(dealCounts, H)
        # _ = get!(dealCounts, H, 0);
        # dealCounts[H] += 1;

    end

    return dealCounts;

end

generateAllHands() = [[]] |> addSuit |> addSuit |> addSuit |> addSuit





function buildDB(hands, probs)
    # nHands = length(hands);

    handIndexDict = Dict{handType, Tuple{Int64, Int64}}()
    playHandIndexDict = Dict{NTuple{4, Int64}, Int64}()

    handProbabilities = []
    discards = discardType[]
    playHands = NTuple{4, Int64}[]
    playCounts = Int64[]
    marginRegrets = Float64[]
    greedyRegrets = Float64[]
    fastRegrets = Float64[]
    marginProfile = Float64[]
    greedyProfile = Float64[]
    fastProfile = Float64[]

    nRow = 1

    @showprogress 1 for (nHand, hand) in enumerate(hands)

        D = getDiscards(hand)
        nDiscards = length(D)

        handIndexDict[Tuple(Tuple.(hand))] = (nRow, nRow + nDiscards -1)

        p = probs[nHand]

        append!(handProbabilities, [p for d in D])
        append!(discards, D)
        pHands = [getPlayHand(hand, d) for d in D]
        append!(playHands, pHands)
        append!(playCounts, zeros(Int64, nDiscards))
        append!(marginRegrets, zeros(Float64, nDiscards))
        append!(greedyRegrets, zeros(Float64, nDiscards))
        append!(fastRegrets, zeros(Float64, nDiscards))
        append!(marginProfile, [1/nDiscards for d in D])
        append!(greedyProfile, [1/nDiscards for d in D])
        append!(fastProfile, [1/nDiscards for d in D])

        for (ii, pHand) in enumerate(pHands)
            playHandIndexDict[pHand] = nRow + ii - 1
        end


        nRow += nDiscards    

    end

    df = DataFrame(handProbability = handProbabilities,
                    discard = discards,
                    playHand = playHands,
                    playCountDealer = playCounts,
                    marginRegretDealer = marginRegrets,
                    greedyRegretDealer = greedyRegrets,
                    fastRegretDealer = fastRegrets,
                    marginProfileDealer = marginProfile,
                    greedyProfileDealer = greedyProfile,
                    fastProfileDealer = fastProfile,
                    playCountPone = playCounts,
                    marginRegretPone = marginRegrets,
                    greedyRegretPone = greedyRegrets,
                    fastRegretPone = fastRegrets,
                    marginProfilePone = marginProfile,
                    greedyProfilePone = greedyProfile,
                    fastProfilePone = fastProfile)

    return (df, handIndexDict, playHandIndexDict)
end





struct FlatPack
    n_v::Vector{Int8};
    n_l::Vector{Int8};
    flatv::Vector{Int8};
    flatl::Vector{Int16};
end
Base.show(io::IO, fp::FlatPack) = print(io, "Flat Pack (", length(fp.n_v), " nodes)");

function packflat(ft::FlatTree)
    n_v = Int8[]
    n_l = Int8[]
    flatv = Int8[]
    flatl = Int16[]
    for ii in 1:length(ft)
        push!(n_v, length(ft.values[ii]))
        push!(n_l, length(ft.links[ii]))
        append!(flatv, ft.values[ii])
        append!(flatl, ft.links[ii])
    end
    return FlatPack(n_v, n_l, flatv, flatl)
end
function packflat(::Nothing)
    return nothing
end

function unpackflat(fp::FlatPack)    
    V = Vector{Union{NTuple{4, Int8}, NTuple{3, Int8}, NTuple{2, Int8}, NTuple{1, Int8}, Tuple{}}}()
    L = Vector{Union{NTuple{4, Int16}, NTuple{3, Int16}, NTuple{2, Int16}, NTuple{1, Int16}, Tuple{}}}()
    iv = 1
    il = 1
    for (jj, n) in enumerate(fp.n_v)
        push!(V, Tuple(fp.flatv[iv:(iv+n-1)]))
        if fp.n_l[jj] > 0
            push!(L, Tuple(fp.flatl[il:(il+fp.n_l[jj]-1)]))
        else
            push!(L, ())
        end
        il += fp.n_l[jj]
        iv += n
    end
    return FlatTree(Tuple(V), Tuple(L));
end
function unpackflat(::Nothing)
    return nothing
end


function gettree(h1::hType, h2::hType)
    return M[phID[h1], phID[h2]]
end





# function gettree(h1::hType, h2::hType)
#     i1 = phID[h1]
#     i2 = phID[h2]
#     if !isnothing(M[i1, i2])
#         return M[i1, i2]
#     end
#     ps = PlayState(2, [countertovector(h1), countertovector(h2)], Int64[], Int64[], 0, 0, 0, [0,0], PlayState[], 0, 0)
#     solve!(ps)
#     mn = MinimalNode((), ())
#     minimize!(ps, mn)
#     ft =makeflat(mn)
#     M[i1, i2] = ft
#     return ft
# end

function saveM()
    serialize("M.jls", tmap(packflat, M))
    GC.gc()
end

function loadM()
    return tmap(unpackflat, deserialize("M.jls"))
end

function Msize()
    return length([m for m in M if !isnothing(m)])
end


function saveprogress()
    serialize("db.jls", db)
    serialize("phDealerProbs.jls", phDealerProbs)
    serialize("phPoneProbs.jls", phPoneProbs)
    serialize("results.jls", results)
end

function loadprogress()
    global db = deserialize("db.jls")
    global phDealerProbs = deserialize("phDealerProbs.jls")
    global phPoneProbs = deserialize("phPoneProbs.jls")
    global results = deserialize("results.jls")
end



function logresult(h1cards::Vector{Card}, h2cards::Vector{Card}, turncard::Card,
                    d1cards::Vector{Card}, d2cards::Vector{Card}, playscores::Vector{Int64}, showscores::Vector{Int64},
                    resultlock::ReentrantLock)
    
    lock(resultlock)
    try
        push!(results,
                OrderedDict(
                    "ndeal" => nresults+1, "h1" => h1cards, "h2" => h2cards, "turn" => turncard, 
                    "d1" => d1cards, "d2" => d2cards, "play" => playscores, "show" => showscores
                )
            )
        global nresults += 1
    finally
        unlock(resultlock)
    end

end


