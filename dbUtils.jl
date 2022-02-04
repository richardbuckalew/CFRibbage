using DataFrames, Serialization

nSuits = 4;
ranks = 1:13;
handSize = 6;
minMaxSuitSize = Int(ceil(handSize / nSuits));

CC = collect.([combinations(ranks, ii) for ii in 0:handSize]);
function newSuits(H)

    nSuit = length(H) + 1;
    (nSuit > 1) ? (cardsleft = handSize - sum(length.(H))) : (cardsleft = handSize);
    (cardsleft == 0) && (return []);

    NS = [];

    if nSuit == 1
        m = minMaxSuitSize;
        M = handSize;
    elseif 1 < nSuit < nSuits
        m = Int(ceil(cardsleft / (nSuits - nSuit + 1)));
        M = cardsleft;
    else
        m = cardsleft;
        M = cardsleft;
    end

    for suitSize in m:M
        append!(NS, CC[suitSize+1]);
    end

    return NS;

end

function addSuit(HH)


    newHH = Set([]);

    for H in HH
        NS = newSuits(H);
        (length(NS) == 0) && push!(newHH, copy(H));
        for ns in NS
            G = copy(H);
            push!(G, ns);
            sort!(sort!(G), by = length);
            push!(newHH, G);
        end
    end

    return newHH

end

function getDiscards(H) ## Discard format: A tuple of tuples, one for each suit in the canonical form of the hand. 
                        ## Respects canonical suit order on a per-hand basis

    D = [];

    for (i1, i2) in combinations(1:4,2)
        (isempty(H[i1]) || isempty(H[i2])) && continue;

        for (c1, c2) in product(H[i1], H[i2])

            d = [[] ,[], [], []];
            d[i1] = [c1];
            d[i2] = [c2];

            # push!(D, d);
            push!(D, Tuple(Tuple.(d)));

        end
    end

    for s in unique(H)
        isempty(s) && continue;

        ii = findfirst(Ref(s) .== H);
        (length(s) < 2) && continue;
        for (c1, c2) in combinations(s, 2)
            d = [[], [], [], []];
            d[ii] = [c1, c2];

            # push!(D, d);
            push!(D, Tuple(Tuple.(d)));
        end
    end

    return D

end

function getPlayHand(h, d)
    return  Tuple(sort(vcat([Array(setdiff(h[ii], d[ii])) for ii in 1:4]...)));
end

function buildHandData()

    hData = Vector{HandData}(undef, length(allHands));
    hIndices = Dict{Any, Int64}();
    phIndices = Dict{Any, Vector{NTuple{2,Int64}}}();

    for (ii, H) in enumerate(allHands)

        D = getDiscards(H);
        nd = length(D);

        PH = Vector{Any}(undef, nd);
        for (jj,d) in enumerate(D)
            PH[jj] = makeDiscard(H, d);
            if haskey(phIndices, PH[jj])
                push!(phIndices[PH[jj]], (ii, jj));
            else
                phIndices[PH[jj]] = [(ii, jj)];
            end
        end

        hData[ii] = HandData(handProbabilities[ii], D, PH, zeros(Int64,nd), zeros(Float64,nd), zeros(Float64,nd));
        hIndices[H] = ii;

    end

    return (hData, hIndices, phIndices);
end

function dealAllHands()

    # dealCounts = Dict{Any, Int64}();
    dealCounts = counter(Vector{Vector{Int64}})
    deck = standardDeck;


    for comb in combinations(deck, 6)

        (H, sp) = canonicalize(comb);

        inc!(dealCounts, H);
        # _ = get!(dealCounts, H, 0);
        # dealCounts[H] += 1;

    end

    return dealCounts;

end

generateAllHands() = [[]] |> addSuit |> addSuit |> addSuit |> addSuit;

handType = Union{
    Tuple{Tuple{Int64, Int64, Int64, Int64, Int64, Int64}, Tuple{}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64, Int64, Int64}, Tuple{Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64, Int64}, Tuple{Int64, Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64, Int64}, Tuple{Int64}, Tuple{Int64}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64}, Tuple{Int64, Int64, Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64}, Tuple{Int64, Int64}, Tuple{Int64}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64}, Tuple{Int64}, Tuple{Int64}, Tuple{Int64}},
    Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}, Tuple{Int64, Int64}, Tuple{}},
    Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}, Tuple{Int64}, Tuple{Int64}}
}
discardType = Union{
    Tuple{Tuple{Int64, Int64}, Tuple{}, Tuple{}, Tuple{}},
    Tuple{Tuple{}, Tuple{Int64, Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{}, Tuple{}, Tuple{Int64, Int64}, Tuple{}},
    Tuple{Tuple{}, Tuple{}, Tuple{}, Tuple{Int64, Int64}},
    Tuple{Tuple{Int64}, Tuple{Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64}, Tuple{}, Tuple{Int64}, Tuple{}},
    Tuple{Tuple{Int64}, Tuple{}, Tuple{}, Tuple{Int64}},
    Tuple{Tuple{}, Tuple{Int64}, Tuple{Int64}, Tuple{}},
    Tuple{Tuple{}, Tuple{Int64}, Tuple{}, Tuple{Int64}},
    Tuple{Tuple{}, Tuple{}, Tuple{Int64}, Tuple{Int64}},
}

function buildDB(hands, probs)
    # nHands = length(hands);

    handIndexDict = Dict{handType, Tuple{Int64, Int64}}();
    playHandIndexDict = Dict{NTuple{4, Int64}, Int64}();

    handProbabilities = [];
    discards = discardType[];
    playHands = NTuple{4, Int64}[];
    playCounts = Int64[];
    marginRegrets = Float64[];
    greedyRegrets = Float64[];
    fastRegrets = Float64[];
    marginProfile = Float64[];
    greedyProfile = Float64[];
    fastProfile = Float64[];

    nRow = 1;

    @showprogress 1 for (nHand, hand) in enumerate(hands)

        D = getDiscards(hand);
        nDiscards = length(D);

        handIndexDict[Tuple(Tuple.(hand))] = (nRow, nRow + nDiscards -1);

        p = probs[nHand];

        append!(handProbabilities, [p for d in D]);
        append!(discards, D);
        pHands = [getPlayHand(hand, d) for d in D];
        append!(playHands, pHands);
        append!(playCounts, zeros(Int64, nDiscards));
        append!(marginRegrets, zeros(Float64, nDiscards));
        append!(greedyRegrets, zeros(Float64, nDiscards));
        append!(fastRegrets, zeros(Float64, nDiscards));
        append!(marginProfile, [1/nDiscards for d in D]);
        append!(greedyProfile, [1/nDiscards for d in D]);
        append!(fastProfile, [1/nDiscards for d in D]);

        for (ii, pHand) in enumerate(pHands)
            playHandIndexDict[pHand] = nRow + ii - 1;
        end


        nRow += nDiscards;       

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
                    fastProfilePone = fastProfile);

    return (df, handIndexDict, playHandIndexDict);
end



# (@isdefined allHands) || (allHands = deserialize("allHands.jls"));
# (@isdefined allHandProbabilities) || (allHandProbabilities = deserialize("allHandProbabilities.jls"));

# (db, hID, pHID) = buildDB(allHands, allHandProbabilities);
# serialize("database.jls", (db, hID, pHID));







