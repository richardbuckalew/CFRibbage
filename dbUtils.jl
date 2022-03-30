using ProgressMeter

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

function dealAllHands(deck)

    handCounts = counter(handType)

    display("Dealing & Canonicalizing hands...")
    @showprogress 1 for comb in combinations(deck, 6)
        (H, sp) = canonicalize(comb)
        inc!(handCounts, H)
    end
    return handCounts;

end



function buildDB(deck)

    handCounts = dealAllHands(deck)
    nHands = sum(values(handCounts))

    prob = Float64[]
    discard = discardType[]
    playhand = NTuple{4, Int}[]
    dealerplaycount = Int64[]
    poneplaycount = Int64[]
    dealerregret = Float64[]
    poneregret = Float64[]
    dealerprofile = Float64[]
    poneprofile = Float64[]
    dealerplayprob = Float64[]
    poneplayprob = Float64[]

    hRows = Dict{handType, NTuple{2, Int}}()
    phRows = Dict{hType, Vector{Int}}()
    allPH = Vector{hType}()
    phID = Dict{hType, Int}()

    n = 0
    display("Building data structures...")
    @showprogress 1 for h in keys(handCounts)

        D = getDiscards(h)
        nd = length(D)

        hRows[h] = (n + 1, n + nd)

        for d in D

            push!(discard, d)
            ph = getPlayHand(h, d)
            push!(playhand, ph)
            phc = counter(ph)
            if phc in keys(phRows)
                push!(phRows[phc], n+1)
            else
                phRows[phc] = [n+1]
                push!(allPH, counter(ph))
            end

            push!(prob, handCounts[h] / nHands)
            push!(dealerplaycount, 0)
            push!(poneplaycount, 0)
            push!(dealerregret, 0.0)
            push!(poneregret, 0.0)
            push!(dealerprofile, 1 / nd)
            push!(poneprofile, 1 / nd)
            push!(dealerplayprob, prob[end] * dealerprofile[end])
            push!(poneplayprob, prob[end] * poneprofile[end])

            n += 1


        end
    end


    db = DataFrame(prob=prob, discard=discard, playhand=playhand, 
                   dealerplaycount=dealerplaycount, dealerregret=dealerregret, dealerprofile=dealerprofile, dealerplayprob=dealerplayprob, 
                   poneplaycount=poneplaycount, poneregret=poneregret, poneprofile=poneprofile, poneplayprob=poneplayprob)


    for (ii, phc) in enumerate(allPH)
        phID[phc] = ii
    end

    phDealerProbs = Dict{hType, Float64}()
    phPoneProbs = Dict{hType, Float64}()
    for phc in allPH
        phDealerProbs[phc] = sum(dealerplayprob[phRows[phc]])
        phPoneProbs[phc] = phDealerProbs[phc]
    end

    return (db, hRows, phRows, phID, allPH, phDealerProbs, phPoneProbs)

end

function buildM(phID, allPH)

    n = length(allPH)
    M = Matrix{Union{FlatTree, Nothing}}(nothing, n, n)

    @showprogress 1 for h1 in allPH
        ph1 = countertovector(h1)
        @floop for h2 in allPH
            ph2 = countertovector(h2)

            if any(values(counter(vcat(ph1, ph2))) .> 4)
                continue   
            end

            i1 = phID[h1]
            i2 = phID[h2]

            ps = PlayState(2, [ph1, ph2], Int64[], Int64[], 0, 0, 0, [0,0], PlayState[], 0, 0)
            solve!(ps)
            mn = MinimalNode((), ())
            minimize!(ps, mn)
            ft = makeflat(mn)
            M[i1, i2] = ft

        end
    end
    
    return M
end


function init_environment()

    display("Building database... ")
    (db, hRows, phRows, phID, allPH, phDealerProbs, phPoneProbs) = buildDB(standardDeck)
    display("Precaching trees... ")
    M = buildM(phID, allPH)


    display("Serializing... ")
    serialize("db.jls", db)
    serialize("hRows.jls", hRows)
    serialize("phDealerProbs.jls", phDealerProbs)
    serialize("phPoneProbs.jls", phPoneProbs)
    serialize("M.jls", tmap(packflat, M))
    serialize("allPH.jls", allPH)
    serialize("phID.jls", phID)
    serialize("phRows.jls", phRows)
    
    display("Ready to train.")

    ## This function is run-once. For training, many of these should load as const in global scope.
    

end


function reset_environment()

    db = deserialize("db.jls")
    phDealerProbs = deserialize("phDealerProbs.jls")
    phPoneProbs = deserialize("phPoneProbs.jls")

    for R in values(hRows)

        db[R[1]:R[2], :dealerplaycount] .= 0
        db[R[1]:R[2], :poneplaycount] .= 0
        db[R[1]:R[2], :dealerregret] .= 0.0
        db[R[1]:R[2], :poneregret] .= 0.0
        db[R[1]:R[2], :dealerprofile] .= 1 / (R[2] - R[1] + 1)
        db[R[1]:R[2], :poneprofile] .= 1 / (R[2] - R[1] + 1)
        db[R[1]:R[2], :dealerplayprob] .= db[R[1]:R[2], :prob] ./ (R[2] - R[1] + 1)
        db[R[1]:R[2], :poneplayprob] .= db[R[1]:R[2], :prob] ./ (R[2] - R[1] + 1)

    end

    for ph in allPH
        phDealerProbs[ph] = sum(db[phRows[ph], :dealerplayprob])
        phPoneProbs[ph] = sum(db[phRows[ph], :poneplayprob])
    end

    serialize("db.jls", db)
    serialize("phDealerProbs.jls", phDealerProbs)
    serialize("phPoneProbs.jls", phPoneProbs)

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
    i1 = phID[h1]
    i2 = phID[h2]
    if !isnothing(M[i1, i2])
        return M[i1, i2]
    end
    ps = PlayState(2, [countertovector(h1), countertovector(h2)], Int64[], Int64[], 0, 0, 0, [0,0], PlayState[], 0, 0)
    solve!(ps)
    mn = MinimalNode((), ())
    minimize!(ps, mn)
    ft =makeflat(mn)
    M[i1, i2] = ft
    return ft
end

function saveM()
    serialize("M.jls", tmap(packflat, M))
end

function loadM()
    return tmap(unpackflat, deserialize("M.jls"))
end

function Msize()
    return length([m for m in M if !isnothing(m)])
end




