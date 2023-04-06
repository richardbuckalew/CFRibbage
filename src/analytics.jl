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


using StatsBase, JSON, DataStructures, Plots, StatsPlots




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








function load_tdata(n::Int)

  filename = "data/tData_" * string(n) * ".json"
  open(filename) do f
    D = JSON.parse(f)
    TD = TrainingData(D["n_dealt"], D["dt"], D["dealt_dealer"], D["dealt_pone"], D["discards_dealer"], D["discards_pone"],
                      D["scores"], D["deltas_dealer"], D["deltas_pone"])
  end
end

function load_sStats(n::Int)

  filename = "data/sStats_" * string(n) * ".json"
  open(filename) do f
    D = JSON.parse(f)
    cd = D["coverage_dealer"]
    cp = D["coverage_pone"]
    hcd = D["hand_counts_dealer"]
    hcp = D["hand_counts_pone"]
    add = D["active_discards_dealer"]
    adp = D["active_discards_pone"]
    hpm = D["Hprob_max"]
    hphd = D["HpHist_dealer"]
    hphp = D["HpHist_pone"]

    hcd = Dict(parse(Int, k) => v for (k,v) in pairs(hcd))
    hcp = Dict(parse(Int, k) => v for (k,v) in pairs(hcp))
    add = Dict(parse(Int, k) => v for (k,v) in pairs(add))
    adp = Dict(parse(Int, k) => v for (k,v) in pairs(adp))

    SS = SummaryStats(cd, cp, hcd, hcp, add, adp, hpm, Tuple(hphd), Tuple(hphp))
  end
end



function find_trained_hands(tol=1e-6)
  # n = 100
  n = 2550    # MAGIC NUMBER: THERE ARE 2550 tData files at time of writing
  n_deal = 0
  learned_dealer = DefaultDict{Int, Vector{Bool}}(Vector{Bool})
  learned_pone = DefaultDict{Int, Vector{Bool}}(Vector{Bool})
  for i in 1:n
    td = load_tdata(i)

    # DEVELOP A CRITERION THAT SAYS: THIS HAND DOESN'T NEED TO BE TRAINED ANY more
    # EG 40 CONSECUTIVE ZEROS. SAVE ONES AND ZEROS IS ENOUGH; WHEN WE DEAL THE HAND DID IT CHANGE?
    # IF SO, A ONE. IF NOT, A ZERO. A TERMINAL STRING OF ZEROS INDICATES CONVERGENCE.


    for n_hand in 1:td.n_dealt
      n_deal += 1                           # absolute # of deal, from 1 to 200 million or whatever
      push!(learned_dealer[td.dealt_dealer[n_hand]], td.deltas_dealer[n_hand] >= tol)
      push!(learned_pone[td.dealt_pone[n_hand]], td.deltas_pone[n_hand] >= tol)
    end
  end

  skiplist_dealer = []
  skiplist_pone = []

  for (hid, learned) in learned_dealer
    (length(learned) < 20) && continue
    any(learned[end-20+1:end]) && continue
    push!(skiplist_dealer, hid)
  end
  for (hid, learned) in learned_pone
    (length(learned) < 20) && continue
    any(learned[end-20+1:end]) && continue
    push!(skiplist_pone, hid)
  end

  return (skiplist_dealer, skiplist_pone)

end


struct TrainingReport
  N_dealt::Vector{Int}
  delta_data::Matrix{Float64}
  zero_fraction::Vector{Float64}
  play_counts::Vector{Int}
end

function training_report()


  db = deserialize("data/db.jls")

  n_hands = length(db.allh)


  n_trainingsets = 3550    # MAGIC NUMBER: THERE ARE 3550 DATA FILES AT TIME OF writing
  # n_trainingsets = 10

  N_dealt = zeros(Int, n_trainingsets)
  delta_data = zeros(n_trainingsets, 5)  # 5%, 25%, 50%, 75%, 95%
  zero_fraction = zeros(n_trainingsets)
  play_counts = zeros(Int, n_hands)



  for n_set in 1:n_trainingsets
    display(n_set)

    TD = load_tdata(n_set)

    if n_set == 1
      N_dealt[n_set] = TD.n_dealt
    else
      N_dealt[n_set] = N_dealt[n_set-1] + TD.n_dealt
    end

    nonzero_deltas = vcat(filter(!isapprox(0.0), TD.deltas_dealer), filter(!isapprox(0.0), TD.deltas_pone))
    n_zero_deltas = 2*TD.n_dealt - length(nonzero_deltas)

    delta_data[n_set, :] = percentile.(Ref(nonzero_deltas), [5, 25, 50, 75, 95])

    # delta_data[n_set, :] = getfield.(Ref(summarystats(nonzero_deltas)), [:mean, :min, :q25, :median, :q75, :max])
    zero_fraction[n_set] = n_zero_deltas / (2 * TD.n_dealt)

    for hid in TD.dealt_dealer
      play_counts[hid] += 1
    end    

  end

  TR = TrainingReport(N_dealt, delta_data, zero_fraction, play_counts)
  serialize("data/training_report.jls", TR)

end





function plot_training_report(TR::TrainingReport)

  delta_plot = plot(TR.N_dealt, TR.delta_data[:, 1:end], 
    color = [:black :black :black], 
    lw = [1 2 3 2 1], ls = [:dot :dash :solid :dash :dot], 
    yaxis = :log, 
    label = ["5th percentile" "25th percentile" "median" "75th percentile" "95th percentile"], 
    legend = :bottomleft,
    ylim = (1e-4, 1))

  play_counts_plot = histogram(TR.play_counts, bins = 200, legend = :false)



    return (delta_plot, play_counts_plot)

end



TR = deserialize("data/training_report.jls")
(delta_plot, play_counts_plot) = plot_training_report(TR)

sStats = load_sStats(3550)

AD = merge(sStats.active_discards_dealer, sStats.active_discards_pone)
active_discards_plot = bar(AD, yaxis = :log, legend = false)

hph_x = sStats.HpHist_dealer[1]
hph_y = sStats.HpHist_dealer[2] + sStats.HpHist_pone[2]

play_probs_plot = bar(hph_x[1:end-1][hph_y .> 0], hph_y[hph_y .> 0], yaxis = :log, legend = false)


plot(delta_plot, play_counts_plot, play_probs_plot, active_discards_plot, layout = (2,2), size = (1200, 800))









