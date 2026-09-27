"""
    run_dda_matrix(samples; device="cpu", num_cores=1, parallelism=:threads, flavors=("ST",), ...)

Run the pure Julia DDA engine on a `samples × channels` matrix. Channel and
pair indices are 1-based. `device` accepts `"cpu"`, `"cuda"`, or `"cuda:N"`.
CUDA.jl is loaded only when a CUDA device is requested. SY evaluates
consecutive channel pairs `(ch₁, ch₂), (ch₃, ch₄), …`; an odd trailing
channel is left unpaired.

CPU window batches run concurrently when `num_cores > 1`. `parallelism=:threads`
uses available Julia threads; `:processes` creates temporary local workers.
Counts are capped at the CPU count and available work. A single CPU window
uses threaded channel/variant solves instead. CUDA keeps its batched
solver and requires `num_cores=1` and `parallelism=:threads`.
"""
function run_dda_matrix(
    samples::AbstractMatrix{<:Real};
    device::AbstractString="cpu",
    num_cores::Integer=1,
    parallelism::Symbol=:threads,
    channels=nothing,
    flavors=("ST",),
    window_length::Int=NATIVE_WINDOW_LENGTH,
    window_step::Int=NATIVE_WINDOW_STEP,
    delays=DDADefaults.DELAYS,
    model_terms=DDADefaults.MODEL_PARAMS,
    derivative_points::Int=DDADefaults.DERIVATIVE_POINTS,
    order::Int=DDADefaults.POLYNOMIAL_ORDER,
    nr_tau::Int=DDADefaults.NUM_TAU,
    ct_channel_pairs=nothing,
    cd_channel_pairs=nothing,
    ct_window_length=nothing,
    ct_window_step=nothing,
    channel_labels=nothing,
    start::Real=0.0,
    stop=nothing,
    normalization::AbstractString="zscore",
    nr_exclude::Int=10,
    derivative_step::Int=1,
)::NativeDDAResult
    num_cores >= 1 || throw(ArgumentError("num_cores must be positive"))
    parallelism in (:threads, :processes) || throw(ArgumentError(
        "parallelism must be :threads or :processes",
    ))
    first(_parse_device(device)) == :cpu || (num_cores == 1 && parallelism == :threads) || throw(ArgumentError(
        "num_cores and parallelism configure CPU execution; CUDA uses its batched solver",
    ))
    ctx = _dda_context(
        samples;
        device=device,
        channels=channels,
        flavors=flavors,
        window_length=window_length,
        window_step=window_step,
        delays=delays,
        model_terms=model_terms,
        derivative_points=derivative_points,
        order=order,
        nr_tau=nr_tau,
        ct_channel_pairs=ct_channel_pairs,
        cd_channel_pairs=cd_channel_pairs,
        ct_window_length=ct_window_length,
        ct_window_step=ct_window_step,
        channel_labels=channel_labels,
        start=start,
        stop=stop,
        normalization=normalization,
        nr_exclude=nr_exclude,
        derivative_step=derivative_step,
    )
    _dda_execute!(ctx, num_cores, parallelism)
    return _dda_results(ctx)
end

function _dda_context(
    samples;
    device,
    channels,
    flavors,
    window_length,
    window_step,
    delays,
    model_terms,
    derivative_points,
    order,
    nr_tau,
    ct_channel_pairs,
    cd_channel_pairs,
    ct_window_length,
    ct_window_step,
    channel_labels,
    start,
    stop,
    normalization,
    nr_exclude,
    derivative_step,
)
    data = _validate_samples(samples)
    row_count, channel_count = size(data)
    labels = _normalize_channel_labels(channel_labels, channel_count)
    selected_channels = _normalize_channels(channels, channel_count)
    enabled = Set(_normalize_flavor(String(flavor)) for flavor in flavors)
    isempty(enabled) && error("At least one DDA flavor must be enabled")
    unsupported = setdiff(enabled, Set(["ST", "CT", "CD", "DE", "SY"]))
    isempty(unsupported) || error("Unsupported Julia DDA flavor(s): $(join(sort(collect(unsupported)), ", "))")
    _parse_device(device)

    model = _build_model_spec(;
        window_length,
        window_step,
        delays,
        model_terms,
        derivative_points,
        order,
        nr_tau,
    )
    bounds_start, bounds_length = _analysis_bounds(start, stop, row_count)
    marker = model.window_length + model.max_delay + 2model.derivative_points
    required_rows = max(marker - 1, 0)
    bounds_length >= required_rows || error(
        "Selected range has $bounds_length samples but DDA requires at least $required_rows " *
        "samples (window_length + 2*derivative_points + max(delay) - 1)",
    )
    window_count = 1 + div(bounds_length - required_rows, model.window_step)
    markers = Float64[
        bounds_start + (window - 1) * model.window_step + marker
        for window in 1:window_count
    ]

    ct_groups = _resolve_ct_groups(
        selected_channels,
        ct_channel_pairs,
        ct_window_length,
        ct_window_step,
        channel_count,
    )
    de_groups = _sliding_groups(
        selected_channels,
        ct_window_length,
        ct_window_step,
    )
    cd_pairs = _resolve_cd_pairs(selected_channels, cd_channel_pairs, channel_count)
    sy_pairs = _sy_pairs(selected_channels)

    enabled_st = "ST" in enabled
    enabled_ct = "CT" in enabled
    enabled_cd = "CD" in enabled
    enabled_de = "DE" in enabled
    enabled_sy = "SY" in enabled && !isempty(sy_pairs)
    any((enabled_st, enabled_ct, enabled_cd, enabled_de, enabled_sy)) || error(
        "No DDA flavors are enabled for the selected channels",
    )

    st_matrix = enabled_st ? fill(NaN, length(selected_channels), window_count) : nothing
    ct_matrix = enabled_ct ? fill(NaN, length(ct_groups), window_count) : nothing
    cd_matrix = enabled_cd ? fill(NaN, length(cd_pairs), window_count) : nothing
    de_matrix = enabled_de ? fill(NaN, length(de_groups), window_count) : nothing
    sy_matrix = enabled_sy ? fill(NaN, length(sy_pairs), window_count) : nothing
    analysis_channels = _analysis_channels(selected_channels, ct_groups, de_groups, cd_pairs)

    jobs_per_window =
        ((enabled_st || enabled_cd || enabled_de) ? length(analysis_channels) : 0) +
        (enabled_ct ? length(ct_groups) : 0) +
        (enabled_de ? length(de_groups) : 0) +
        (enabled_cd ? length(cd_pairs) : 0) +
        (enabled_sy ? 2length(sy_pairs) : 0)
    windows_per_batch = max(1, min(32, div(2048, max(jobs_per_window, 1))))
    env_batch = get(ENV, "DDA_JL_BATCH", nothing)
    env_batch === nothing || (windows_per_batch = max(1, parse(Int, env_batch)))

    return (
        data=data,
        device=device,
        labels=labels,
        selected_channels=selected_channels,
        channel_count=channel_count,
        model=model,
        bounds_start=bounds_start,
        window_count=window_count,
        markers=markers,
        normalization=String(normalization),
        nr_exclude=nr_exclude,
        derivative_step=derivative_step,
        enabled_st=enabled_st,
        enabled_ct=enabled_ct,
        enabled_cd=enabled_cd,
        enabled_de=enabled_de,
        enabled_sy=enabled_sy,
        ct_groups=ct_groups,
        de_groups=de_groups,
        cd_pairs=cd_pairs,
        sy_pairs=sy_pairs,
        analysis_channels=analysis_channels,
        windows_per_batch=windows_per_batch,
        feature_count=length(model.primary_terms),
        st_matrix=st_matrix,
        ct_matrix=ct_matrix,
        cd_matrix=cd_matrix,
        de_matrix=de_matrix,
        sy_matrix=sy_matrix,
    )
end

function _dda_window_batch(ctx, first_window)
    last_window = min(first_window + ctx.windows_per_batch - 1, ctx.window_count)
    problems = RegressionProblem[]
    references = WindowProblemRefs[]
    model = ctx.model

    for window in first_window:last_window
        prepared = _prepare_window(
            ctx.data,
            ctx.bounds_start,
            model,
            window,
            ctx.normalization,
            ctx.nr_exclude,
            ctx.derivative_step,
        )
        st_refs = zeros(Int, ctx.channel_count)
        if ctx.enabled_st || ctx.enabled_cd || ctx.enabled_de
            for channel in ctx.analysis_channels
                st_refs[channel] = _push_problem!(
                    problems,
                    _group_problem(prepared, [channel], model.primary_terms, model.window_length),
                )
            end
        end
        ct_refs = ctx.enabled_ct ? Int[
            _push_problem!(
                problems,
                _group_problem(prepared, group, model.primary_terms, model.window_length),
            ) for group in ctx.ct_groups
        ] : Int[]
        de_refs = ctx.enabled_de ? Int[
            _push_problem!(
                problems,
                _group_problem(prepared, group, model.primary_terms, model.window_length),
            ) for group in ctx.de_groups
        ] : Int[]
        cd_refs = ctx.enabled_cd ? Int[
            _push_problem!(
                problems,
                _directed_problem(
                    prepared,
                    target,
                    source,
                    target,
                    model.primary_terms,
                    model.window_length,
                ),
            ) for (target, source) in ctx.cd_pairs
        ] : Int[]
        sy_forward = ctx.enabled_sy ? Int[
            _push_problem!(
                problems,
                _directed_problem(
                    prepared,
                    left,
                    right,
                    right,
                    model.primary_terms,
                    model.window_length,
                ),
            ) for (left, right) in ctx.sy_pairs
        ] : Int[]
        sy_reverse = ctx.enabled_sy ? Int[
            _push_problem!(
                problems,
                _directed_problem(
                    prepared,
                    right,
                    left,
                    left,
                    model.primary_terms,
                    model.window_length,
                ),
            ) for (left, right) in ctx.sy_pairs
        ] : Int[]
        push!(references, WindowProblemRefs(st_refs, ct_refs, de_refs, cd_refs, sy_forward, sy_reverse))
    end

    return problems, references
end

function _dda_unpack!(ctx, solutions, references, first_window)
    feature_count = ctx.feature_count
    for (offset, refs) in enumerate(references)
        window = first_window + offset - 1
        if ctx.st_matrix !== nothing
            for (row, channel) in enumerate(ctx.selected_channels)
                block = _block(solutions, refs.st[channel], feature_count)
                !isempty(block.coefficients) && (ctx.st_matrix[row, window] = block.coefficients[1])
            end
        end
        if ctx.ct_matrix !== nothing
            for row in eachindex(ctx.ct_groups)
                block = _block(solutions, refs.ct[row], feature_count)
                !isempty(block.coefficients) && (ctx.ct_matrix[row, window] = block.coefficients[1])
            end
        end
        if ctx.de_matrix !== nothing
            for row in eachindex(ctx.de_groups)
                block = _block(solutions, refs.de[row], feature_count)
                ctx.de_matrix[row, window] = _de_value(
                    ctx.de_groups[row], refs.st, solutions, block.rmse, feature_count
                )
            end
        end
        if ctx.cd_matrix !== nothing
            for (row, (target, _)) in enumerate(ctx.cd_pairs)
                baseline = _block(solutions, refs.st[target], feature_count).rmse
                directed = _block(solutions, refs.cd[row], 2feature_count).rmse
                ctx.cd_matrix[row, window] = _causal_improvement(baseline, directed)
            end
        end
        if ctx.sy_matrix !== nothing
            for row in eachindex(ctx.sy_pairs)
                forward = _block(solutions, refs.sy_forward[row], 2feature_count).rmse
                reverse = _block(solutions, refs.sy_reverse[row], 2feature_count).rmse
                ctx.sy_matrix[row, window] = _synchronization_value(forward, reverse)
            end
        end
    end
    return nothing
end

function _dda_results(ctx)
    results = NativeFlavorResult[]
    ctx.st_matrix !== nothing && push!(results, NativeFlavorResult(
        "ST", "Single Timeseries (ST)", ctx.st_matrix,
        _channel_labels(ctx.labels, ctx.selected_channels), ctx.markers,
    ))
    ctx.ct_matrix !== nothing && push!(results, NativeFlavorResult(
        "CT", "Cross-Timeseries (CT)", ctx.ct_matrix,
        _group_labels(ctx.labels, ctx.ct_groups, "-"), ctx.markers,
    ))
    ctx.cd_matrix !== nothing && push!(results, NativeFlavorResult(
        "CD", "Cross-Dynamical (CD)", ctx.cd_matrix,
        _pair_labels(ctx.labels, ctx.cd_pairs, " <- "), ctx.markers,
    ))
    ctx.de_matrix !== nothing && push!(results, NativeFlavorResult(
        "DE", "Dynamical Ergodicity (DE)", ctx.de_matrix,
        _group_labels(ctx.labels, ctx.de_groups, "-"), ctx.markers,
    ))
    ctx.sy_matrix !== nothing && push!(results, NativeFlavorResult(
        "SY", "Synchronization (SY)", ctx.sy_matrix,
        _pair_labels(ctx.labels, ctx.sy_pairs, " <-> "), ctx.markers,
    ))
    return NativeDDAResult(results, ctx.markers, ctx.labels)
end

function _sliding_groups(channels::Vector{Int}, window_length, window_step)::Vector{Vector{Int}}
    length_value = window_length === nothing ? length(channels) : Int(window_length)
    step = window_step === nothing ? max(length_value, 1) : max(Int(window_step), 1)
    (length_value <= 0 || length(channels) < length_value) && return Vector{Vector{Int}}()
    groups = Vector{Vector{Int}}()
    start_idx = 1
    while start_idx + length_value - 1 <= length(channels)
        push!(groups, copy(channels[start_idx:(start_idx + length_value - 1)]))
        start_idx += step
    end
    return groups
end

function _normalize_groups(groups, channel_count::Int)::Vector{Vector{Int}}
    normalized = Vector{Vector{Int}}()
    for group in groups
        values = Int[group...]
        isempty(values) && error("Channel groups must not be empty")
        all(channel -> 1 <= channel <= channel_count, values) || error(
            "Channel group indices must be in 1:$channel_count",
        )
        push!(normalized, values)
    end
    return normalized
end

function _resolve_ct_groups(channels, pairs, window_length, window_step, channel_count)
    pairs !== nothing && !isempty(pairs) && return _normalize_groups(pairs, channel_count)
    return _sliding_groups(channels, window_length, window_step)
end

function _resolve_cd_pairs(channels, pairs, channel_count)::Vector{Tuple{Int,Int}}
    if pairs !== nothing && !isempty(pairs)
        groups = _normalize_groups(pairs, channel_count)
        all(group -> length(group) == 2, groups) || error("CD pairs must contain two channels")
        return Tuple{Int,Int}[(group[1], group[2]) for group in groups]
    end
    return Tuple{Int,Int}[
        (target, source) for target in channels for source in channels if target != source
    ]
end

"""SY evaluates consecutive channel pairs `(ch₁, ch₂), (ch₃, ch₄), …`;
an odd trailing channel is left unpaired."""
function _sy_pairs(channels::Vector{Int})::Vector{Tuple{Int,Int}}
    return Tuple{Int,Int}[
        (channels[index], channels[index + 1]) for index in 1:2:(length(channels) - 1)
    ]
end

function _analysis_channels(st_channels, ct_groups, de_groups, cd_pairs)::Vector{Int}
    channels = Set(st_channels)
    foreach(group -> union!(channels, group), ct_groups)
    foreach(group -> union!(channels, group), de_groups)
    foreach(pair -> union!(channels, pair), cd_pairs)
    return sort(collect(channels))
end

_channel_labels(labels, channels) = String[labels[channel] for channel in channels]
_group_labels(labels, groups, joiner) = String[
    join(_channel_labels(labels, group), joiner) for group in groups
]
_pair_labels(labels, pairs, joiner) = String[
    "$(labels[left])$joiner$(labels[right])" for (left, right) in pairs
]
