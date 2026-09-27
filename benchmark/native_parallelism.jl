#!/usr/bin/env julia

using Dates
using DelayDifferentialAnalysis
using LinearAlgebra
using Printf
using Random
using SHA
using Statistics
using TOML

const Native = DelayDifferentialAnalysis.NativeDDA
const CASES = [
    (name="small_st", samples=2048, channels=4, flavors=["ST"]),
    (name="single_window", samples=2063, channels=64, flavors=["ST"]),
    (name="wide_st", samples=16384, channels=64, flavors=["ST"]),
    (name="all_flavors", samples=8192, channels=12, flavors=["ST", "CT", "CD", "DE", "SY"]),
    (name="long_all", samples=32768, channels=16, flavors=["ST", "CT", "CD", "DE", "SY"]),
]

function option(args, name, default)
    index = findfirst(==(name), args)
    index === nothing && return default
    index < length(args) || error("Missing value after $name")
    return args[index + 1]
end

function samples_for(case)
    rng = MersenneTwister(1729)
    return [
        sin((0.011 + 0.003channel) * row + 0.17channel) +
        0.3cos(0.031row - 0.1channel) + 0.01randn(rng)
        for row in 1:case.samples, channel in 1:case.channels
    ]
end

function parameters(case)
    return (
        device="cpu", channels=nothing, flavors=case.flavors,
        window_length=case.name == "single_window" ? 2048 : 128,
        window_step=64, delays=[2, 6],
        model_terms=[1, 2, 4], derivative_points=3, order=3, nr_tau=2,
        ct_channel_pairs=nothing, cd_channel_pairs=nothing,
        ct_window_length=2, ct_window_step=1, channel_labels=nothing,
        start=0.0, stop=nothing, normalization="zscore", nr_exclude=10,
        derivative_step=1,
    )
end

# This is the original staged engine, retained as the benchmark reference.
function legacy_solve(problems)
    solutions = Vector{Native.SolvedBlock}(undef, length(problems))
    Threads.@threads :dynamic for index in eachindex(problems)
        solutions[index] = Native._solve_cpu(problems[index])
    end
    return solutions
end

function reference_run(samples, parameters; threaded=false)
    ctx = Native._dda_context(samples; parameters...)
    preparation = solve = assembly = 0.0
    for first_window in 1:ctx.windows_per_batch:ctx.window_count
        preparation += @elapsed problems, references = Native._dda_window_batch(ctx, first_window)
        solve += @elapsed solutions = threaded ? legacy_solve(problems) : Native._solve_cpu(problems)
        assembly += @elapsed Native._dda_unpack!(ctx, solutions, references, first_window)
    end
    return Native._dda_results(ctx), (preparation=preparation, solve=solve, assembly=assembly)
end

function compare_results(actual, expected)
    actual.window_markers == expected.window_markers || error("Window markers changed")
    actual.channel_labels == expected.channel_labels || error("Channel labels changed")
    [f.id for f in actual.flavors] == [f.id for f in expected.flavors] || error("Flavor order changed")
    difference = 0.0
    exact = true
    for (a, b) in zip(actual.flavors, expected.flavors)
        a.row_labels == b.row_labels || error("Flavor row labels changed")
        size(a.matrix) == size(b.matrix) || error("Output shape changed")
        exact &= isequal(a.matrix, b.matrix)
        for (x, y) in zip(a.matrix, b.matrix)
            isequal(x, y) && continue
            isfinite(x) && isfinite(y) || error("Nonfinite output mismatch")
            isapprox(x, y; atol=1e-10, rtol=1e-9) || error("Output mismatch: $x vs $y")
            difference = max(difference, abs(x - y))
        end
    end
    return difference, exact
end

function measure(samples, parameters, mode, workers)
    GC.gc()
    if mode == "serial"
        return @timed first(reference_run(samples, parameters))
    elseif mode == "legacy"
        return @timed first(reference_run(samples, parameters; threaded=true))
    end
    return @timed run_dda_matrix(
        samples; parameters..., num_cores=workers, parallelism=Symbol(mode),
    )
end

function write_paired_speedups(rows, output)
    rng = MersenneTwister(1729)
    open(joinpath(output, "paired_speedups.csv"), "w") do io
        println(io, "case,baseline,mode,workers,repeats,median_paired_speedup,bootstrap_95_low,bootstrap_95_high,win_fraction")
        for case in unique(r.case for r in rows), baseline in ("serial", "legacy")
            base = Dict(r.repeat => r.seconds for r in rows if r.case == case && r.mode == baseline)
            isempty(base) && continue
            for (mode, count) in unique((r.mode, r.workers) for r in rows if r.case == case)
                mode == baseline && continue
                ratios = [base[r.repeat] / r.seconds for r in rows
                    if r.case == case && r.mode == mode && r.workers == count]
                boot = [median(rand(rng, ratios, length(ratios))) for _ in 1:2000]
                low, high = quantile(boot, [0.025, 0.975])
                println(io, join((case, baseline, mode, count, length(ratios),
                    median(ratios), low, high, mean(ratios .> 1)), ','))
            end
        end
    end
end

function main(args=ARGS)
    output = option(args, "--output", "results/native_julia_parallelism")
    repeats = parse(Int, option(args, "--repeats", "5"))
    workers = parse.(Int, split(option(args, "--workers", "1,2,4,8,16"), ','))
    modes = split(option(args, "--modes", "threads,processes"), ',')
    selected = split(option(args, "--cases", join(getproperty.(CASES, :name), ',')), ',')
    cases = filter(case -> case.name in selected, CASES)
    length(cases) == length(selected) || error("Unknown or duplicate case")
    repeats > 0 && all(>(0), workers) || error("Repeats and workers must be positive")
    all(mode -> mode in ("legacy", "threads", "processes"), modes) || error("Unknown mode")
    "threads" in modes && maximum(workers) > Threads.nthreads() && error(
        "Start Julia with --threads=$(maximum(workers)) or request fewer thread workers",
    )
    maximum(workers) <= Sys.CPU_THREADS || error("Requested workers exceed the CPU count")
    ispath(output) && error("Output already exists: $output")
    mkpath(output)
    BLAS.set_num_threads(1)
    metadata = Dict(
        "timestamp" => string(now()), "host" => readchomp(`hostname`),
        "julia" => string(VERSION), "julia_threads" => Threads.nthreads(),
        "logical_cpus" => Sys.CPU_THREADS, "cpu" => first(Sys.cpu_info()).model,
        "blas_threads" => BLAS.get_num_threads(), "blas" => string(BLAS.get_config()),
        "seed" => 1729, "repeats" => repeats, "workers" => workers,
        "modes" => modes, "cases" => selected, "arguments" => args,
        "timing_scope" => "Complete call, including process startup, transfer, and teardown",
        "allocated_bytes_scope" => "Parent process only; excludes child-process allocations",
        "benchmark_sha256" => bytes2hex(sha256(read(@__FILE__))),
    )
    source_root = dirname(pathof(DelayDifferentialAnalysis))
    metadata["source_sha256"] = Dict(
        relpath(joinpath(dir, file), source_root) => bytes2hex(sha256(read(joinpath(dir, file))))
        for (dir, _, files) in walkdir(source_root) for file in files if endswith(file, ".jl")
    )
    open(io -> TOML.print(io, metadata), joinpath(output, "metadata.toml"), "w")
    rows = NamedTuple[]
    rng = MersenneTwister(1729)
    configurations = [(mode="serial", workers=1)]
    for mode in modes, count in (mode == "legacy" ? [Threads.nthreads()] : workers)
        push!(configurations, (mode=mode, workers=count))
    end
    open(joinpath(output, "timings.csv"), "w") do io
        println(io, "case,mode,workers,repeat,samples,channels,windows,seconds,allocated_bytes,gc_seconds,max_abs_difference,exact")
        open(joinpath(output, "stages.csv"), "w") do stages
            println(stages, "case,preparation_seconds,solve_seconds,assembly_seconds")
            for case in cases
                samples, params = samples_for(case), parameters(case)
                expected, _ = reference_run(samples, params)
                _, times = reference_run(samples, params)
                println(stages, join((case.name, times.preparation, times.solve, times.assembly), ','))
                flush(stages)
                # Compile every path before timing; each multiprocessing call still
                # creates and removes its own processes in the measured interval.
                for config in configurations
                    measured = measure(samples, params, config.mode, config.workers)
                    compare_results(measured.value, expected)
                end
                for repeat in 1:repeats, config in shuffle(rng, configurations)
                    measured = measure(samples, params, config.mode, config.workers)
                    difference, exact = compare_results(measured.value, expected)
                    row = (case=case.name, mode=config.mode, workers=config.workers,
                        repeat=repeat, samples=case.samples, channels=case.channels,
                        windows=length(expected.window_markers), seconds=measured.time,
                        allocated_bytes=measured.bytes, gc_seconds=measured.gctime,
                        max_abs_difference=difference, exact=exact)
                    push!(rows, row)
                    println(io, join(values(row), ','))
                    flush(io)
                    @printf("%s %s/%d repeat %d: %.4fs, difference %.3g\n",
                        case.name, config.mode, config.workers, repeat, measured.time, difference)
                    flush(stdout)
                end
            end
        end
    end
    open(joinpath(output, "summary.csv"), "w") do io
        println(io, "case,mode,workers,median_seconds,min_seconds,max_seconds,speedup_vs_serial,efficiency,max_abs_difference,exact")
        for case in cases
            serial = median(r.seconds for r in rows if r.case == case.name && r.mode == "serial")
            configs = unique((r.mode, r.workers) for r in rows if r.case == case.name)
            for (mode, count) in configs
                group = filter(r -> r.case == case.name && r.mode == mode && r.workers == count, rows)
                times = getproperty.(group, :seconds)
                speedup = serial / median(times)
                println(io, join((case.name, mode, count, median(times), minimum(times), maximum(times),
                    speedup, speedup / count, maximum(r.max_abs_difference for r in group),
                    all(r.exact for r in group)), ','))
            end
        end
    end
    write_paired_speedups(rows, output)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
