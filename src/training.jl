"""
  training.jl contains the structures and algorithms that underly the CFR loop. This includes information modeling
and the bayesian inference update step. There are some important constants defined here for efficiency as well.
"""

import DataStructures


#   For development, and potentially for training, too, we define an information model that does not contain enough
# to support the bayesian update step. When using this model, probabilities will always be based on the base assumption
# of the current strategy profile, rather than updating in response to opponent plays.
#
#   I suspect that this version trains almost as well. It will be an interesting question to investigate whether
# there is a meaninful difference between the two approaches. Both will converge, so the questions are 1) do they converge
# to the same ϵ-solution? and 2) is the computational effeciency tradeoff worth it?


struct InformationModel_Base
    include::HType              # counts of the ranks known to be in the unknown hand.
    exclude::HType              # counts of the ranks known not to be in the unknown hand.
    model::BitVector            # Indices correspond to HID. model[ix] is True if HID^-1[ix] is admissable.
    probs::Vector{Float64}      # The probabilities I assign to each admissible hand
    pointers::Vector{Int64}     # Pointers into the FlatTrees corresponding to the hands in the model. These represent the 
                                #   game state we would be in if opponent has the corresponding hand H
    trees::Vector{Union{FlatTree, Nothing}}     # The row or column from M that corresponds to our hand
end

function InformationModel_Base()
    return InformationModel_Base(
                Accumulator{Int64, Int64}(),
                Accumulator{Int64, Int64}(),
                trues(n_H),
                fill(1.0 / n_H, n_H),
                fill(1, n_H),
                fill(FlatTree(), n_H)
                )
end

function Base.show(io::IO, IM::InformationModel_Base)
    print(io, "IM_Base. Known include: ", IM.include, ";  Known exclude: ", IM.exclude, "\n")
    T = hcat(IM.model, IM.probs)
    print(T)
end


function init_model!(IM::InformationModel_Base, known_cards::HType, whichplayer::Int64, H::HType, M, HID)
    for (r, m) in known_cards
        newExclude!(IM, r, m)
    end
    for ix in eachindex(IM.model)
        if whichplayer == 1
            (M[HID[H], ix] == Nothing) && (IM.model[ix] = false)
        elseif whichplayer == 2
            (M[ix, HID[H]] == Nothing) && (IM.model[ix] = false)
        end
    end
end

#NOTE: The following assumes that allH is ordered according to HID, so that HID[allH] == 1:n_H
"Set the probabilities in an information model according to the strategy profile corresponding to Hprobs."
function init_probs!(IM::InformationModel_Base, allH::Vector{HType}, Hprobs::Dict{HType, Float64})
    for ix in eachindex(IM.probs)
        IM.probs[ix] = Hprobs[allH[ix]]
    end
end

"Initialize IM.trees to the correct FlatTrees for our hand H. If whichplayer == 1, then we are dealer."
function init_trees!(IM::InformationModel_Base, whichplayer::Int64, H::HType, M, HID)
    for ix in eachindex(IM.trees)
        if whichplayer == 1
            IM.trees[ix] = M[HID[H], ix]
        elseif whichplayer == 2
            IM.trees[ix] = M[ix, HID[H]]
        end
    end
end


#   A variation of Bayes' Theorem goes: P(B|A) = [ P(A|B)P(B) ] / [ P(A|B)P(B) + P(A|~B)P(~B) ]. Here, B is "opponent makes
# the play they just made", and A is "opponent has a particular hand H". P(B) is the prior; to evaluate P(A|B) we need an 
# approximation of opponent's knowledge, hence the addition of a metamodel to the information model.


struct InformationModel_Bayes
    include::HType              # counts of the ranks known to be in the unknown hand.
    exclude::HType              # counts of the ranks known not to be in the unknown hand.
    meta_include::HType         # counts of the ranks known by my opponent to be in my hand.
    meta_exclude::HType         # counts of the ranks known by my opponent not to be in my hand.
    model::BitVector            # Indices correspond to HID. model[ix] is True if HID^-1[ix] is admissable.
    metamodel::BitMatrix        # Both axes are indexed like HID. metamodel[m,n] is True means: If opponent has hand HID^-1[m],
                                #   then based on my knowledge, they should consider HID^-1[n] admissible. This only makes sense
                                #   for rows of metamodel that correspond to hands admissible according to my model.
                                # Note that for the row of metamodel corresponding to opponent's *actual* hand, my metamodel
                                #   will not be equal model, because opponent has different info than I do. All that's guaranteed
                                #   is that my metamodel row will always be True at the column corresponding to my actual hand
                                #   and opponent's model will be True at the row corresponding to my actual hand.
    p_model::Vector{Float64}    # The probabilities I assign to each admissible hand
    p_metamodel::Vector{Float64}    # The probabilities opponent assigns to each possible hand, according to the metamodel
    pointers_model::Vector{Int64}   # Pointers into the FlatTrees corresponding to the hands in the model. These represent the 
                                    #   game state we would be in if opponent has the corresponding hand H
    pointers_metamodel::Matrix{Int64}   # Pointers into the FlatTrees corresponding to the hands in the metamodel.
end

# TODO: duplicate the constructor, reset, etc. for the bayes version


## Functions for updating and resetting Information Models

function reset!(IM::InformationModel_Base)
    for k in 1:13
        IM.include[k] = 0
        IM.exclude[k] = 0
    end
    fill!(IM.model, true)
    fill!(IM.pointers, 1)
    fill!(IM.probs, 1.0/n_H)
end


function renormalize!(IM::InformationModel_Base)
    IM.probs ./= sum(@view IM.probs[IM.model])
end


"Update the IM's model based on the knowledge that mult new copies of r must be included in opponent's hand."
function newInclude!(IM::InformationModel_Base, r::Int64, mult::Int64 = 1)
    IM.include[r] += mult
    IM.model .&= @view imask[:,IM.include[r],r]         # imask is a global defined in CFRibbage.jl
end


"Update the information model IM based on the knowledge that mult new copies of r must be excluded from opponent's hand."
function newExclude!(IM::InformationModel_Base, r::Int64, mult::Int64 = 1)
    IM.exclude[r] += mult
    IM.model .&= @view emask[:,IM.exclude[r], r]         # emask is a global defined in CFRibbage.jl
end


"""
Update the information model IM in response to an opponent's play when total was playtotal.
This involves: 1) updating model; 2) updating pointers; 3) updating probs.
"""
function opponentPlay!(IM::InformationModel_Base, play::Int64, playtotal::Int64)

    # Update model
    if play == 0   # GO
        for r in 1:(31 - playtotal)
            newExclude!(IM, r, 4 - IM.exclude[r])
        end
    else            # a card was played
        newInclude!(IM, play)
    end

    # Update pointers
    for (ix, tree) in enumerate(IM.trees)
        isnothing(tree) && continue             # no tree to point to
        (IM.model[ix] == false) && continue     # dead pointer

        for (jx, r) in enumerate(tree[IM.pointers[ix]][1])      # look thru the potential plays for opp at this node
            if r == play
                IM.pointers[ix] = tree[IM.pointers[ix]][2][jx]  # once we find the play that happened, jump
                break
            end
        end
    end

    # Update probs
    renormalize!(IM)

end


# TODO: refactor and pull out the for loop common to opponentPlay! and myPlay!
"Update the IM (base type) after I play. I learn nothing, and there's no metamodel, so just update pointers."
function myPlay!(IM::InformationModel_Base, play::Int64)
    for (ix, tree) in enumerate(IM.trees)
        isnothing(tree) && continue
        (IM.model[ix] == false) && continue
        (length(tree[IM.pointers[ix]][2]) == 0) && continue     # natural end of the tree

        for (jx, r) in enumerate(tree[IM.pointers[ix]][1])
            if r == play
                IM.pointers[ix] = tree[IM.pointers[ix]][2][jx]
                break
            end
        end
    end
end


"Choose the best play based on IM, as player whichplayer."
function getPlay(IM::InformationModel_Base, whichplayer::Int64)
    (whichplayer == 1) ? (whichmod = 1) : (whichmod = -1)       # p2 prefers negatives, but we want to compare positives.
    totalEVs = fill(0.0, 13)
    bestEV = -99.0
    bestplay = -1    
    for ix in eachindex(IM.model)
        IM.model[ix] || continue        # only consider live options
        (length(IM.trees[ix][IM.pointers[ix]][1]) == 0) && (return 0)
        (length(IM.trees[ix][IM.pointers[ix]][1]) == 1) && (return IM.trees[ix][IM.pointers[ix]][1][1])
        for jx in eachindex(IM.trees[ix][IM.pointers[ix]][1])
            totalEVs[IM.trees[ix][IM.pointers[ix]][1][jx]] += whichmod * IM.probs[ix] * IM.trees[ix][IM.pointers[ix]][3][jx]
            if totalEVs[IM.trees[ix][IM.pointers[ix]][1][jx]] > bestEV
                bestEV = totalEVs[IM.trees[ix][IM.pointers[ix]][1][jx]]
                bestplay = IM.trees[ix][IM.pointers[ix]][1][jx]
            end
        end
    end
    return bestplay
end






"Get the result of one play hand"
function playHand(H1::HType, H2::HType, IM1::InformationModel_Base, IM2::InformationModel_Base)
    total = 0
    whoseturn = 2
    firstgo = false
    history = Int64[]

    # println(c2v(H1), " vs ", c2v(H2))

    while (any(values(H1) .> 0) || any(values(H2) .> 0))

        if whoseturn == 1
            if any(values(H1) .> 0)
                p = Int64(getPlay(IM1, 1))
                myPlay!(IM1, p)
            else
                p = 0
            end
            push!(history, p)
            opponentPlay!(IM2, p, total)
            if p == 0
                if firstgo
                    total = 0
                    firstgo = false
                    println("2x go")
                else
                    firstgo = true
                end
            else
                H1[p] -= 1
                total += cardvalues[p]
            end
        elseif whoseturn == 2
            if any(values(H2) .> 0)
                p = Int64(getPlay(IM2, 2))
                myPlay!(IM2, p)
            else
                p = 0
            end
            push!(history, p)
            opponentPlay!(IM1, p, total)
            if p == 0
                if firstgo
                    total = 0
                    firstgo = false
                    println("2x go")
                else
                    firstgo = true
                end
            else
                H2[p] -= 1
                total += cardvalues[p]
            end
        end

        # println(c2v(H1), " vs ", c2v(H2))
        # println("  history: ", history)    
        # println("  total: ", total)    

        # flush(stdout)

        # sleep(0.5)

        whoseturn = 3-whoseturn

    end
    return history
end







## BIG NOTE: The include and exclude masks and methods assume all four suits exist in the deck. 
## IF YOU TEST ON A DECK WITH FEWER THAN 4 SUITS, YOU MUST CHANGE THE MAGIC NUMBER 4 THAT OCCURS 23 LINES DOWN.

"Generate a bit mask of size n_H x 4 x 13. imask[m,n,r] is true if HID^-1[m] contains at least n copies of rank r."
function generateIncludeMask()
    imask = falses(n_H, 4, 13)
    for r in 1:13
        for k in 1:4
            for H in allH
                Hid = HID[H]
                (H[r] >= k) && (imask[Hid, k, r] = true)
            end
        end
    end
    return imask
end

"Generate a bit mask of size n_H x 4 x 13. emask[m,n,r] is true if HID^-1[m] contains at most 4-n copies of rank r."
function generateExcludeMask()
    emask = falses(n_H, 4, 13)
    for r in 1:13
        for k in 1:4
            for H in allH
                Hid = HID[H]
                (H[r] <= (4-k)) && (emask[Hid, k, r] = true)
            end
        end
    end
    return emask
end






