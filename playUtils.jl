
(@isdefined pSS) || (const pSS = (2, 6, 12));
(@isdefined run3diffs) || (const run3diffs = [[1, 1], [-1, -1], [2, -1], [-2, 1], [-1, 2], [1, -2]]);

mutable struct PerfectPlayState
    owner::Int64;
    hands::Vector{Vector{Int64}};
    history::Vector{Int64};
    diffs::Vector{Int64};
    total::Int64;
    pairlength::Int64;
    runlength::Int64;
    scores::Vector{Int64};
    children::Vector{PerfectPlayState};
    value::Int64;
    bestplay::Int64;
end
function Base.show(io::IO, ps::PerfectPlayState)
    print(io, "PlayState (owner ", ps.owner, ")");
end
function Base.show(io::IO, ::MIME"text/plain", ps::PerfectPlayState)
    print(io, "PlayState (p", ps.owner, ")\n");
    print(io, "  Hands:   ", ps.hands, "\n");
    print(io, "  History: ", ps.history, " (total: ", ps.total, ")\n");
    print(io, "  Scores:  ", ps.scores, "\n");
    print(io, "  Value:   ", ps.value, ", Best Play: ", ps.bestplay);
end



function solvePlayPerfect(hands)
    ps = PerfectPlayState(2, hands, Int64[], Int64[], 0, 0, 0, [0,0], PerfectPlayState[], 0, 0);
    return perfectsolve!(ps);
end

function matchupPlayHands(hands)
    C = counter(vcat(hands...));
    counts = collect(values(C));
    # display(C);
    (maximum(counts) > 4) ? (return nothing) : (return solvePlayPerfect([hands...]));
end



function perfectsolve!(ps::PerfectPlayState)


    if all(isempty.(ps.hands))
        newscores = ps.scores;
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

        newowner = 3 - ps.owner;
        newhands = deepcopy(ps.hands); deleteat!(newhands[ps.owner], ii);
        newhistory = vcat(ps.history, [c]);
        (length(ps.history) > 0) ? (newdiffs = vcat(ps.diffs, [c-ps.history[end]])) : (newdiffs = Int64[]);

        newscores = copy(ps.scores);
        if length(ps.history) > 0
            if c == ps.history[end]
                newpairlength = ps.pairlength + 1;
                newscores[ps.owner] += pSS[newpairlength];
            else
                newpairlength = 0;
            end
        else
            newpairlength = 0;
        end
        if length(newdiffs) > 1
            if ps.runlength == 0
                if newdiffs[(end-1):end] in run3diffs
                    newrunlength = ps.runlength + 1;
                    newscores[ps.owner] += newrunlength + 2;
                else
                    newrunlength = 0;
                end
            else
                currentrun = ps.history[(end-1-ps.runlength):end];
                if (c == minimum(currentrun) - 1) || (c == maximum(currentrun) + 1)
                    newrunlength = ps.runlength + 1;
                    newscores[ps.owner] += newrunlength + 2;
                else
                    newrunlength = 0;
                end
            end
        else
            newrunlength = 0;
        end
        (newtotal == 15) && (newscores[ps.owner] += 2);
        (newtotal == 31) && (newscores[ps.owner] += 1);

        child = PerfectPlayState(newowner, newhands, newhistory, newdiffs, newtotal, 
                newpairlength, newrunlength, newscores, PerfectPlayState[], 0, 0)
        push!(ps.children, child);
        push!(childvalues, perfectsolve!(child));
    end

    if length(childvalues) > 0
        (ps.owner == 1) ? ((m, mi) = findmax(childvalues)) : ((m, mi) = findmin(childvalues));
        ps.value = m;
        ps.bestplay = candidates[mi];
    else
        newowner = 3 - ps.owner;
        newhands = deepcopy(ps.hands);
        newhistory = vcat(ps.history, [0]);
        newchildren = PerfectPlayState[];
        if ps.history[end] == 0     # this is a double GO, so reset
            newdiffs = Int64[];
            newtotal = 0;
            newpairlength = 0;
            newrunlength = 0;
            newscores = ps.scores;
        else                        # this is the first GO.
            newdiffs = ps.diffs;
            newtotal = ps.total;
            newpairlength = ps.pairlength;
            newrunlength = ps.runlength;
            newscores = copy(ps.scores); newscores[newowner] += 1;    # other player gets a point
        end
        push!(ps.children,
                PerfectPlayState(newowner, newhands, newhistory, newdiffs, newtotal, 
                newpairlength, newrunlength, newscores, newchildren, 0, 0)
        );
        ps.value = perfectsolve!(ps.children[1]);
        ps.bestplay = 0;
    end

    return ps.value;

end

function buildPerfectSolveMatrix(phID)

    PH = [[x...] for x in sort(collect(keys(phID)))];
    handPairs = collect(product(PH, deepcopy(PH)));
    return tmap(matchupPlayHands, handPairs);

end



function testperfectsolve()
    (h1Cards, h2Cards, turnCard) = dealHands();

    h1 = [c.rank for c in h1Cards];
    h2 = [c.rank for c in h2Cards];
    hands = [h1[1:4], h2[1:4]];
    history = Int64[];
    total = 0;
    diffs = Int64[];
    pairlength = 0;
    runlength = 0;
    scores = [0,0];
    children = PerfectPlayState[];
    value = 0;
    bestplay = 0;

    ps = PerfectPlayState(2, hands, history, diffs, total, pairlength, runlength,
            scores, children, value, bestplay);
    perfectsolve!(ps);

    return ps
end




