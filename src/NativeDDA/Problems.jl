function _evaluate_term(
    prepared::PreparedWindow,
    channel::Int,
    sample::Int,
    delays::Vector{Int},
)::Float64
    product = 1.0
    for delay in delays
        row = sample + max(prepared.max_delay - delay, 0)
        value = prepared.shifted[row, channel]
        isnan(value) && return NaN
        product *= value
    end
    return product
end

# Tasks may migrate between threads. Each owns its scratch; design and targets
# must stay distinct even when their dimensions coincide (a one-term model).
@inline function _scratch_matrix(role::Symbol, dims)::Matrix{Float64}
    store = get!(task_local_storage(), :dda_problem_scratch) do
        Dict{Tuple{Symbol,Tuple{Int,Int}},Matrix{Float64}}()
    end
    return get!(store, (role, dims)) do
        Matrix{Float64}(undef, dims)
    end
end

function _group_problem(
    prepared::PreparedWindow,
    channels::Vector{Int},
    terms::Vector{Vector{Int}},
    window_length::Int,
)::Union{RegressionProblem,Nothing}
    total_rows = length(channels) * window_length
    total_rows == 0 && return nothing
    n_terms = length(terms)
    design = _scratch_matrix(:design, (total_rows, n_terms))
    target = _scratch_matrix(:target, (total_rows, 1))
    tvec = @view target[:, 1]
    valid_rows = 0

    for channel in channels, sample in 1:window_length
        target_value = prepared.derivative[channel, sample]
        isnan(target_value) && continue
        ok = true
        @inbounds for c in eachindex(terms)
            v = _evaluate_term(prepared, channel, sample, terms[c])
            isnan(v) && (ok = false; break)
            design[valid_rows + 1, c] = v
        end
        ok || continue
        valid_rows += 1
        tvec[valid_rows] = target_value
    end
    valid_rows / total_rows >= 0.60 || return nothing
    values = copy(@view tvec[1:valid_rows])
    return RegressionProblem(copy(@view design[1:valid_rows, :]), values, copy(values))
end

function _directed_problem(
    prepared::PreparedWindow,
    primary_channel::Int,
    secondary_channel::Int,
    response_channel::Int,
    terms::Vector{Vector{Int}},
    window_length::Int,
)::Union{RegressionProblem,Nothing}
    feature_count = 2length(terms)
    design = _scratch_matrix(:design, (window_length, feature_count))
    targets = _scratch_matrix(:target, (window_length, 2))
    fit_vec = @view targets[:, 1]
    res_vec = @view targets[:, 2]
    valid_rows = 0
    n_sec = length(terms)

    for sample in 1:window_length
        fit_value = prepared.derivative[primary_channel, sample]
        isnan(fit_value) && continue
        ok = true
        @inbounds for c in eachindex(terms)
            v = _evaluate_term(prepared, secondary_channel, sample, terms[c])
            isnan(v) && (ok = false; break)
            design[valid_rows + 1, c] = v
        end
        ok || continue
        @inbounds for c in eachindex(terms)
            v = _evaluate_term(prepared, primary_channel, sample, terms[c])
            isnan(v) && (ok = false; break)
            design[valid_rows + 1, n_sec + c] = v
        end
        ok || continue
        valid_rows += 1
        fit_vec[valid_rows] = fit_value
        res_vec[valid_rows] = prepared.derivative[response_channel, sample]
    end
    valid_rows / window_length >= 0.60 || return nothing
    return RegressionProblem(
        copy(@view design[1:valid_rows, :]),
        copy(@view fit_vec[1:valid_rows]),
        copy(@view res_vec[1:valid_rows]),
    )
end

function _push_problem!(
    problems::Vector{RegressionProblem},
    problem::Union{RegressionProblem,Nothing},
)::Int
    problem === nothing && return 0
    push!(problems, problem)
    return length(problems)
end

function _block(solutions::Vector{SolvedBlock}, reference::Int, feature_count::Int)::SolvedBlock
    return iszero(reference) ? _nan_block(feature_count) : solutions[reference]
end

function _de_value(
    channels::Vector{Int},
    st_references::Vector{Int},
    solutions::Vector{SolvedBlock},
    ct_rmse::Float64,
    feature_count::Int,
)::Float64
    (isempty(channels) || isnan(ct_rmse) || iszero(ct_rmse)) && return NaN
    baseline = mean(
        _block(solutions, st_references[channel], feature_count).rmse
        for channel in channels
    )
    return isnan(baseline) ? NaN : abs(baseline / ct_rmse - 1.0)
end

function _causal_improvement(baseline::Float64, causal::Float64)::Float64
    return isnan(baseline) || isnan(causal) ? NaN : baseline - causal
end

function _synchronization_value(forward::Float64, reverse::Float64)::Float64
    (isnan(forward) || isnan(reverse) || iszero(forward) || iszero(reverse)) && return NaN
    return reverse / forward - forward / reverse
end
