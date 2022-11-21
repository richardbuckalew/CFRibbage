"""
analytics.jl holds the tools for logging progress and processing it.
  A web dashboard (probably plotly.dash; I'll consider Genie too) will monitor training progress. So logs
need to be threadsafe. To achieve this, we'll log regularly and have the progress monitor lag behind by one
full log file.

  The most obvious format for a log file is a snapshot of the current state. These are large,
not cross-platform (without work), and not suited for cloud storage, so we will also log summary statistics 
in json form.

  Training will proceed in segments; during a segment the results of dealt hands will be stored in a TrainingData
object and used to generate a SummaryStats object at the end of the segment. The real-time data stored includes:
  - IDs of hands dealt to each player (from hID)
  - id of the discard chosen by each player (integer index within hRows)
  - net scores achieved for each hand dealt
  - the (normed) change in strategy for each hand dealt to each player
"""


using StatsBase




struct TrainingData
    n_dealt::Int64                              # of hands in this training set
    dt::Float64                                 # cpu time spent on this set
    dealt_dealer::Vector{Int64}                 # a list of ids from hID dealt to the dealer
    dealt_pone::Vector{Int64}
    discards_dealer::Vector{Int64}              # a list of the relative indices of each discard played
    discards_pone::Vector{Int64}
    scores::Vector{Int64}                       # the net score (value) of each hand played
    deltas_dealer::Vector{Float64}              # the ∞-normed deltas of each hand
    deltas_pone::Vector{Float64}
end
function Base.show(io::IO, ::MIME"text/plain", td::TrainingData)
    println(io, "Training Data (", td.n_dealt, " hands in ", round(td.dt, digits=2), "s):")
    println(io, "  hIDs dealt:")
    println(io, "    dealer: ", td.dealt_dealer)
    println(io, "      pone: ", td.dealt_pone)
    println(io, "  Discards chosen:")
    println(io, "    dealer: ", td.discards_dealer)
    println(io, "      pone: ", td.discards_pone)
    println(io, "  Factual scores: ", td.scores)
    println(io, "  ∞ Norm:")
    println(io, "    dealer: ", td.deltas_dealer)
    println(io, "      pone: ", td.deltas_pone)
end

struct SummaryStats
    coverage_dealer::Float64                    # Percentage of all hands seen
    coverage_pone::Float64                      
    hand_counts_dealer::Accumulator{Int64, Int64}      # keys: # of times a hand has been seen; values: number of such hands
    hand_counts_pone::Accumulator{Int64, Int64}
    active_discards_dealer::Accumulator{Int64, Int64}  # keys: # of active discards; values: # of such hands
    active_discards_pone::Accumulator{Int64, Int64}
    Hprob_max::Float64                          # the highest play hand probability
    HpHist_dealer::Tuple{Vector{Float64}, Vector{Int64}}    # histogram of Hprobs. Format: (bin_bounds, counts)
    HpHist_pone::Tuple{Vector{Float64}, Vector{Int64}}
end
function Base.show(io::IO, ::MIME"text/plain", S::SummaryStats)
    print(io, "Summary Stats:\n")
    print(io, "    Coverage: ", (round(S.coverage_dealer, digits=4), round(S.coverage_pone, digits = 4)), "\n")
end


"Create a SummaryStats object from a snapshot of df and related data."
function summarize(db)

    # stats to be gleaned from df
    covered_dealer = 0 
    covered_pone = 0

    hand_counts_dealer = Accumulator{Int64, Int64}()
    hand_counts_pone = Accumulator{Int64, Int64}()

    active_discards_dealer = Accumulator{Int64, Int64}()
    active_discards_pone = Accumulator{Int64, Int64}()

    for rows in values(db.hRows)
        (db.df.dealt_dealer[rows.start] > 0) && (covered_dealer += 1)
        hand_counts_dealer[db.df.dealt_dealer[rows.start]] += 1
        ad_key = count(x -> (x > 1e-6), db.df.profile_dealer[rows])
        active_discards_dealer[ad_key] += 1

        (db.df.dealt_pone[rows.start] > 0) && (covered_pone += 1)
        hand_counts_pone[db.df.dealt_pone[rows.start]] += 1
        ad_key = count(x -> (x > 1e-6), db.df.profile_pone[rows])
        active_discards_pone[ad_key] += 1
    end

    coverage_dealer = covered_dealer / db.n_h
    coverage_pone = covered_pone / db.n_h


    # Hprobs
    Hprob_max = max(maximum(values(db.Hprobs_dealer)), maximum(values(db.Hprobs_pone)))
    edges = LinRange(0.0, Hprob_max, 101)
    hpd = fit(Histogram, collect(values(db.Hprobs_dealer)), edges)
    hpp = fit(Histogram, collect(values(db.Hprobs_pone)), edges)
    HpHist_dealer = (edges, hpd.weights)
    HpHist_pone = (edges, hpp.weights)



    return SummaryStats(coverage_dealer, coverage_pone, hand_counts_dealer, hand_counts_pone,
                         active_discards_dealer, active_discards_pone, Hprob_max, HpHist_dealer, HpHist_pone)

end




