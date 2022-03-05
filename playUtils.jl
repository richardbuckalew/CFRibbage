(@isdefined pSS) || (const pSS = (2, 6, 12));
(@isdefined run3diffs) || (const run3diffs = [[1, 1], [-1, -1], [2, -1], [-2, 1], [-1, 2], [1, -2]]);
renormalize!(v::Vector{Float64}) = (v ./= sum(v));

mutable struct PlayState
    owner::Int64;
    hands::Vector{Vector{Int64}};
    history::Vector{Int64};
    diffs::Vector{Int64};
    total::Int64;
    pairlength::Int64;
    runlength::Int64;
    scores::Vector{Int64};
    children::Vector{PlayState};
    value::Int64;
    bestplay::Int64;
end
function Base.show(io::IO, ps::PlayState)
    print(io, "PlayState (p", ps.owner, "), values: ", [c.value for c in ps.children]);
end
function Base.show(io::IO, ::MIME"text/plain", ps::PlayState)
    print(io, "PlayState (p", ps.owner, ")\n");
    print(io, "  Hands:   ", ps.hands, "\n");
    print(io, "  History: ", ps.history, " (total: ", ps.total, ")\n");
    print(io, "  Scores:  ", ps.scores, "\n");
    print(io, "  Value:   ", ps.value, ", Best Play: ", ps.bestplay);
end
function showtree(ps::PlayState; depth = 0)
    display(" " ^ (depth) * string(ps))
    for c in ps.children
        showtree(c; depth = depth + 3);
    end
end

function solve!(ps::PlayState)


    if all(isempty.(ps.hands))
        newscores = copy(ps.scores);
        newscores[3-ps.owner] += 1;
        ps.value = newscores[1] - newscores[2];
        ps.bestplay = 0;
        return ps.value;
    end

    candidates = Int64[];
    childvalues = Int64[];

    for (ii, c) in enumerate(ps.hands[ps.owner])
        newtotal = cardvalues[c] + ps.total;
        (newtotal > 31) && (continue);
        (c in candidates) && (continue);
        push!(candidates, c);

        
        (length(ps.history) > 0) ? (newdiffs = vcat(ps.diffs, [c-ps.history[end]])) : (newdiffs = Int64[]);

        (s, newpairlength, newrunlength) = scoreplay(c, ps.history, newdiffs, newtotal,
                                                        ps.pairlength, ps.runlength);
        newscores = copy(ps.scores);
        newscores[ps.owner] += s;


        newhands = deepcopy(ps.hands); deleteat!(newhands[ps.owner], ii);

        push!(ps.children, PlayState(3 - ps.owner, newhands, vcat(ps.history, [c]), newdiffs, newtotal, 
                                    newpairlength, newrunlength, newscores, PlayState[], 0, 0));
        push!(childvalues, solve!(ps.children[end]));
    end

    if length(childvalues) > 0
        (ps.owner == 1) ? ((m, mi) = findmax(childvalues)) : ((m, mi) = findmin(childvalues));
        ps.value = m;
        ps.bestplay = candidates[mi];
    else
        if ps.history[end] == 0     # this is a double GO, so reset
            push!(ps.children,
                    PlayState(3 - ps.owner, deepcopy(ps.hands), vcat(ps.history, [0]), Int64[], 0, 
                    0, 0, ps.scores, PlayState[], 0, 0)
            );
        else                        # this is the first GO.
            newscores = copy(ps.scores); newscores[3 - ps.owner] += 1;    # other player gets a point
            push!(ps.children,
                    PlayState(3 - ps.owner, deepcopy(ps.hands), vcat(ps.history, [0]), ps.diffs, ps.total, 
                    ps.pairlength, ps.runlength, newscores, PlayState[], 0, 0)
            );
        end
        ps.value = solve!(ps.children[end]);
        ps.bestplay = 0;
    end

    return ps.value;

end



mutable struct MinimalNode
    childvalues::Union{NTuple{4, Int8}, NTuple{3, Int8}, NTuple{2, Int8}, NTuple{1, Int8}, Tuple{}}
    childnodes::Union{NTuple{4, MinimalNode}, NTuple{3, MinimalNode}, NTuple{2, MinimalNode}, NTuple{1, MinimalNode}, Tuple{}}
end
function Base.show(io::IO, mn::MinimalNode)
    print(io, mn.childvalues)
end
function showtree(mn::MinimalNode; depth = 0)
    display(" " ^ (depth) * string(mn))
    for c in mn.childnodes
        showtree(c; depth = depth + 3)
    end
end

function minimize!(ps::PlayState, mn::MinimalNode)::Nothing
    isempty(ps.children) && (return nothing)
    # (all(length.(ps.hands) .<= 1)) && (return nothing);       ## removed to fix a bug in FlatTree advancement. 
                                                                ## NOT strictly necessary; we can re-add this line if we also
                                                                ## include logic in the play solver that lookas at opponent's hand 
                                                                ## length. If your opponent has no cards, you don't need
                                                                ## the tree any more. Worst case scenario, you just call scoreplay()
                                                                ## twice.
                                                                ## This is a low-priority optimization. It will save a few bytes 
                                                                ## in the FlatTree struct, but I don't think it will speed things
                                                                ## up since there will be just one tree at this point anyway.
    mn.childvalues = Tuple([c.value for c in ps.children])
    mn.childnodes = Tuple([MinimalNode((), ()) for c in ps.children])
    for (ii, c) in enumerate(mn.childnodes)
        minimize!(ps.children[ii], c)
    end
    if isempty(mn.childnodes[1].childvalues)
        mn.childnodes = ()
    end
    return nothing
end




struct FlatTree
    values::Tuple{Vararg{Union{NTuple{4, Int8}, NTuple{3, Int8}, NTuple{2, Int8}, NTuple{1, Int8}, Tuple{}}}}
    links::Tuple{Vararg{Union{NTuple{4, Int16}, NTuple{3, Int16}, NTuple{2, Int16}, NTuple{1, Int16}, Tuple{}}}}
end
Base.length(ft::FlatTree) = length(ft.values)
Base.getindex(ft::FlatTree, i::Int64) = (ft.values[i], ft.links[i])
Base.show(io::IO, ft::FlatTree) = print(io, "Flat Tree (", length(ft), " nodes) with v = ", ft.values[1], " and l = ", ft.links[1])
function advancetree(ft::FlatTree, cindex::Int64)
    return FlatTree(
        Tuple(ft.values[ft[1][2][cindex]:end]),
        Tuple(ft.links[ft[1][2][cindex]:end])
    )
end

function makeflat(mn::MinimalNode)

    Q = [mn];
    V = Vector{Union{NTuple{4, Int8}, NTuple{3, Int8}, NTuple{2, Int8}, NTuple{1, Int8}, Tuple{}}}()
    L = Vector{Union{NTuple{4, Int16}, NTuple{3, Int16}, NTuple{2, Int16}, NTuple{1, Int16}, Tuple{}}}()
    while !isempty(Q)
        s = popfirst!(Q)
        v = Int8[]
        l = Int16[]
        if length(s.childvalues) > 0
            for (ii, c) in enumerate(s.childnodes)
                push!(Q, c)
                push!(v, s.childvalues[ii])
                push!(l, length(Q))
            end
            push!(V, Tuple(v))
            push!(L, Tuple(l))
        end
    end
    return FlatTree(Tuple(V), Tuple(L))

end






mutable struct ModelState       ## later optimization: Store columnwise so it's easy to access model.probs etc.
    hand::hType
    tree::FlatTree
    ptr::Int64
    prob::Float64
end
function advancestate!(ms::ModelState, cindex::Int64)
    ms.ptr += ms.tree.links[ms.ptr][cindex]
end

function optimalplay(candidates::Vector{Int64}, model::Vector{ModelState}, whoseturn::Int64)
    EVs = zeros(Float64, length(candidates));
    for ii in 1:length(candidates)
        # EVs[ii] = sum([ms.tree.values[ms.ptr][ii] for ms in model] .* [ms.prob for ms in model])
        EVs[ii] = dot([ms.tree.values[ms.ptr][ii] for ms in model], [ms.prob for ms in model])
    end
    (whoseturn == 1) ? ((m, mi) = findmax(EVs)) : ((m, mi) = findmin(EVs));
    return (mi, candidates[mi]);
end


function getModels(seen::hType, hid::Int64, owner::Int64)

    models = ModelState[]
    for ph in allPH

        any(values(merge(ph, seen)) .> 4) && continue
        (owner == 1) ? push!(models, ModelState(copy(ph), M[hid, phID[ph]], 1, 1.0)) : push!(models, ModelState(copy(ph), M[phID[ph], hid], 1, 1.0))

    end
    return models

end

function renormalize!(model::Vector{ModelState})
    s = sum([ms.prob for ms in model])
    for ms in model
        ms.prob /= s
    end
end

function naiveplay(hands::Vector{Vector{Int64}}, discards::Vector{Vector{Int64}}, turnrank::Int64)

    whoseturn = 2
    history = Int64[]
    diffs = Int64[]
    total = 0
    scores = [0,0]
    pairlength = 0
    runlength = 0

    played = [counter(Int64[]) for ii in (1, 2)]
    seen = [counter(vcat(hands[ii], discards[ii], [turnrank])) for ii in (1, 2)]

    models = [getModels(seen[ii], phID[counter(hands[ii])], ii) for ii in (1, 2)]

    setinitialprobs!(models[1], 2)
    setinitialprobs!(models[2], 1)


    filter!(ms->(ms.prob > 0.0), models[1])
    filter!(ms->(ms.prob > 0.0), models[2])
    


    while true

        if all(isempty.(hands))
            scores[3-whoseturn] += 1
            break
        end

        # print("p", whoseturn, "'s turn with ", hands[whoseturn], ".\n  total: ", total, "\n  history: ", history, "\n  scores: ", scores, "\n")

        candidates = Int64[]
        handindex = Int64[]
        for (hi, c) in enumerate(hands[whoseturn])
            (cardvalues[c] + total > 31) && continue
            (c in candidates) && continue
            push!(candidates, c)
            push!(handindex, hi)
        end

        if isempty(candidates)
            if history[end] == 0
                diffs = Int64[]
                total = 0
                pairlength = 0
                runlength = 0
            else
                scores[3-whoseturn] += 1
            end
            push!(history, 0)

            for r in 1:(31-total)
                seen[3-whoseturn][r] = 4
            end

            models[3-whoseturn] = [ms for ms in models[3-whoseturn] if all([ms.hand[r] == 0 for r in 1:(31-total)])]
            renormalize!(models[3-whoseturn])
            advancestate!.(models[3-whoseturn], 1)
            advancestate!.(models[whoseturn], 1)



        elseif length(candidates) == 1
            c = candidates[1]
            total += cardvalues[c]
            (length(history) > 0) ? (diffs = vcat(diffs, [c-history[end]])) : (diffs = Int64[])     
            (s, pairlength, runlength) = scoreplay(c, history, diffs, total, pairlength, runlength) ## It is important that diffs has been updated but history has not ##
            scores[whoseturn] += s
            deleteat!(hands[whoseturn], handindex[1])
            push!(history, c) 

            inc!(seen[3-whoseturn], c)
            inc!(played[whoseturn], c)

            # models[3-whoseturn] = [ms for ms in models[3-whoseturn] if ms.hand[c] > 0]
            filter!(ms->(ms.hand[c] > 0), models[3-whoseturn])
            renormalize!(models[3-whoseturn])
            for ms in models[3-whoseturn]
                ci = getcandindex(ms.hand, c)
                if length(hands[3-whoseturn]) > 1
                    advancestate!(ms, ci)
                end
                dec!(ms.hand, c)
            end
            if length(hands[whoseturn]) > 1
                advancestate!.(models[whoseturn], 1)
            end
                                                                    

        else
            (cindex, c) = optimalplay(candidates, models[whoseturn], whoseturn)
            total += cardvalues[c]
            (length(history) > 0) ? (diffs = vcat(diffs, [c-history[end]])) : (diffs = Int64[])     
            (s, pairlength, runlength) = scoreplay(c, history, diffs, total, pairlength, runlength) ## It is important that diffs has been updated but history has not ##
            scores[whoseturn] += s
            deleteat!(hands[whoseturn], handindex[cindex])
            push!(history, c) 

            inc!(seen[3-whoseturn], c)
            inc!(played[whoseturn], c)
            
            # models[3-whoseturn] = [ms for ms in models[3-whoseturn] if ms.hand[c] > 0]
            filter!(ms->(ms.hand[c] > 0), models[3-whoseturn])
            renormalize!(models[3-whoseturn])
            for ms in models[3-whoseturn]
                ci = getcandindex(ms.hand, c)
                try
                    if length(hands[3-whoseturn]) > 1
                        advancestate!(ms, ci)
                    end
                catch
                    @infiltrate
                end
                dec!(ms.hand, c)
            end
            advancestate!.(models[whoseturn], cindex)

        end
        whoseturn = 3 - whoseturn

    end
    return (scores[1] - scores[2], scores, history)

end

function threadednaiveplay(hands::Vector{Vector{Int64}}, discards::Vector{Vector{Int64}}, turnrank::Int64, dealerlock::ReentrantLock, ponelock::ReentrantLock)

    whoseturn = 2
    history = Int64[]
    diffs = Int64[]
    total = 0
    scores = [0,0]
    pairlength = 0
    runlength = 0

    played = [counter(Int64[]) for ii in (1, 2)]
    seen = [counter(vcat(hands[ii], discards[ii], [turnrank])) for ii in (1, 2)]

    models = [getModels(seen[ii], phID[counter(hands[ii])], ii) for ii in (1, 2)]

    lock(dealerlock)
    try
        setinitialprobs!(models[1], 2)
        # setinitialprobs!(models[2], 1)
    finally
        unlock(dealerlock)
    end
    lock(ponelock)
    try
        setinitialprobs!(models[2], 1)
    finally
        unlock(ponelock)
    end


    filter!(ms->(ms.prob > 0.0), models[1])
    filter!(ms->(ms.prob > 0.0), models[2])
    


    while true

        if all(isempty.(hands))
            scores[3-whoseturn] += 1
            break
        end

        # print("p", whoseturn, "'s turn with ", hands[whoseturn], ".\n  total: ", total, "\n  history: ", history, "\n  scores: ", scores, "\n")

        candidates = Int64[]
        handindex = Int64[]
        for (hi, c) in enumerate(hands[whoseturn])
            (cardvalues[c] + total > 31) && continue
            (c in candidates) && continue
            push!(candidates, c)
            push!(handindex, hi)
        end

        if isempty(candidates)
            if history[end] == 0
                diffs = Int64[]
                total = 0
                pairlength = 0
                runlength = 0
            else
                scores[3-whoseturn] += 1
            end
            push!(history, 0)

            for r in 1:(31-total)
                seen[3-whoseturn][r] = 4
            end

            models[3-whoseturn] = [ms for ms in models[3-whoseturn] if all([ms.hand[r] == 0 for r in 1:(31-total)])]
            renormalize!(models[3-whoseturn])
            advancestate!.(models[3-whoseturn], 1)
            advancestate!.(models[whoseturn], 1)



        elseif length(candidates) == 1
            c = candidates[1]
            total += cardvalues[c]
            (length(history) > 0) ? (diffs = vcat(diffs, [c-history[end]])) : (diffs = Int64[])     
            (s, pairlength, runlength) = scoreplay(c, history, diffs, total, pairlength, runlength) ## It is important that diffs has been updated but history has not ##
            scores[whoseturn] += s
            deleteat!(hands[whoseturn], handindex[1])
            push!(history, c) 

            inc!(seen[3-whoseturn], c)
            inc!(played[whoseturn], c)

            # models[3-whoseturn] = [ms for ms in models[3-whoseturn] if ms.hand[c] > 0]
            filter!(ms->(ms.hand[c] > 0), models[3-whoseturn])
            renormalize!(models[3-whoseturn])
            for ms in models[3-whoseturn]
                ci = getcandindex(ms.hand, c)
                if length(hands[3-whoseturn]) > 1
                    advancestate!(ms, ci)
                end
                dec!(ms.hand, c)
            end
            if length(hands[whoseturn]) > 1
                advancestate!.(models[whoseturn], 1)
            end
                                                                    

        else
            (cindex, c) = optimalplay(candidates, models[whoseturn], whoseturn)
            total += cardvalues[c]
            (length(history) > 0) ? (diffs = vcat(diffs, [c-history[end]])) : (diffs = Int64[])     
            (s, pairlength, runlength) = scoreplay(c, history, diffs, total, pairlength, runlength) ## It is important that diffs has been updated but history has not ##
            scores[whoseturn] += s
            deleteat!(hands[whoseturn], handindex[cindex])
            push!(history, c) 

            inc!(seen[3-whoseturn], c)
            inc!(played[whoseturn], c)
            
            # models[3-whoseturn] = [ms for ms in models[3-whoseturn] if ms.hand[c] > 0]
            filter!(ms->(ms.hand[c] > 0), models[3-whoseturn])
            renormalize!(models[3-whoseturn])
            for ms in models[3-whoseturn]
                ci = getcandindex(ms.hand, c)
                try
                    if length(hands[3-whoseturn]) > 1
                        advancestate!(ms, ci)
                    end
                catch
                    @infiltrate
                end
                dec!(ms.hand, c)
            end
            advancestate!.(models[whoseturn], cindex)

        end
        whoseturn = 3 - whoseturn

    end
    return (scores[1] - scores[2], scores, history)

end




function setinitialprobs!(model::Vector{ModelState}, player::Int64)
    for ms in model
        if player == 1
            ms.prob = phDealerProbs[ms.hand]
        else
            ms.prob = phPoneProbs[ms.hand]
        end
    end
end



function issubhand(a::Accumulator{Int64, Int64}, b::Accumulator{Int64, Int64})
    for c in keys(a)
        (a[c] > b[c]) && (return false);
    end
    return true;
end

function excludes(a::Accumulator{Int64, Int64}, b::Accumulator{Int64, Int64})
    for c in keys(a)
        (a[c] + b[c] > 4) && (return false);
    end
    return true;
end


function getPotentialHands(include::Accumulator{Int64, Int64}, exclude::Accumulator{Int64, Int64}, PH::Vector{Accumulator{Int64, Int64}})
    return filter((x -> excludes(x, exclude)), filter((x -> issubhand(include, x)), PH))
end

function getPotentialHandIndices(include::Accumulator{Int64, Int64}, exclude::Accumulator{Int64, Int64}, PH::Vector{Accumulator{Int64, Int64}})
    return findall( (x -> issubhand(include, x)).(PH) .&& (x -> excludes(x, exclude)).(PH) )
end


"""
    optimalplay(candidates, possiblegamestates, probabilities, whoseturn)

Return the optimal play & its e.v. from `candidates`, based on the probability of being in each possible game state 
"""
function optimalplay(candidates::Vector{Int64}, possiblegamestates::Vector{FlatTree}, probabilities::Vector{Float64}, whoseturn::Int64)
    EVs = zeros(Float64, length(candidates));
    for (ii, pgs) in enumerate(possiblegamestates)
        for (jj, _) in enumerate(candidates)
            EVs[jj] += pgs.values[1][jj] * probabilities[ii];
        end
    end
    (whoseturn == 1) ? ((m, mi) = findmax(EVs)) : ((m, mi) = findmin(EVs));
    return (mi, candidates[mi], m);
end


# function getcandindex(h::hType, c::Int64)
#     return findfirst(unique(countertovector(h)) .== c)
# end


function getcandindex(h::hType, c::Int64)::Int64
    v = countertovector(h)
    (v[1] == c) && (return 1)

    cindex = 1
    for (ii, vc) in enumerate(v[2:end])
        (vc == v[ii]) && continue
        cindex += 1
        (vc == c) && (return cindex)
    end

end




function scoreplay(play::Int64, history::Vector{Int64}, diffs::Vector{Int64}, total::Int64,
                    pairlength::Int64, runlength::Int64)
    s = 0;
    if length(history) > 0
        if play == history[end]
            pairlength += 1;
            s += pSS[pairlength];
        else
            pairlength = 0;
        end
    else
        pairlength = 0;
    end
    if length(diffs) > 1
        if runlength == 0
            if diffs[(end-1):end] in run3diffs
                runlength += 1;
                s += runlength+2;
            else
                runlength = 0;
            end
        else
            currentrun = history[(end-1-runlength):end];
            if (play == minimum(currentrun) - 1) || (play == maximum(currentrun) + 1)
                runlength += 1;
                s += runlength+2;
            else
                runlength = 0;
            end
        end
    else
        runlength = 0;
    end
    (total == 15) && (s += 2);
    (total == 31) && (s += 1);
    return (s, pairlength, runlength);
end





function testsolve()
    (h1Cards, h2Cards, turnCard) = dealHands();

    h1 = sort([c.rank for c in h1Cards])
    h2 = sort([c.rank for c in h2Cards])
    hands = [h1[1:4], h2[1:4]]
    history = Int64[]
    total = 0
    diffs = Int64[]
    pairlength = 0
    runlength = 0
    scores = [0,0]
    children = PlayState[]
    value = 0
    bestplay = 0

    ps = PlayState(2, hands, history, diffs, total, pairlength, runlength,
            scores, children, value, bestplay)
    solve!(ps)
    return ps
end

#  ps = testsolve()