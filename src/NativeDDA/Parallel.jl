function _dda_batch!(ctx, first_window; output_window=first_window, num_cores=1)
    problems, references = _dda_window_batch(ctx, first_window)
    # Parallelize whole batches, not nested solves. CUDA still solves on-device.
    solutions = _solve_problems(problems, ctx.device; num_cores)
    _dda_unpack!(ctx, solutions, references, output_window)
    return nothing
end

function _dda_execute!(ctx, num_cores, parallelism)
    limit = parallelism == :threads ? Threads.nthreads() : Sys.CPU_THREADS
    count = Int(min(num_cores, Sys.CPU_THREADS, limit))
    if ctx.window_count == 1 && parallelism == :threads
        # One full-recording window can still have many independent channel fits.
        _dda_batch!(ctx, 1; num_cores=count)
        return nothing
    end
    count = min(count, ctx.window_count)
    ctx = merge(ctx, (windows_per_batch=min(ctx.windows_per_batch, cld(ctx.window_count, count)),))
    if count == 1
        for first_window in 1:ctx.windows_per_batch:ctx.window_count
            _dda_batch!(ctx, first_window)
        end
    elseif parallelism == :threads
        starts = 1:ctx.windows_per_batch:ctx.window_count
        @sync for worker in 1:count
            Threads.@spawn for index in worker:count:length(starts)
                _dda_batch!(ctx, starts[index])
            end
        end
    else
        _dda_processes!(ctx, count)
    end
    return nothing
end

_dda_outputs(ctx) = (; ctx.st_matrix, ctx.ct_matrix, ctx.cd_matrix, ctx.de_matrix, ctx.sy_matrix)

function _dda_process_chunk(ctx, windows)
    outputs = map(_dda_outputs(ctx)) do matrix
        matrix === nothing ? nothing : fill(NaN, size(matrix, 1), length(windows))
    end
    ctx = merge(ctx, outputs, (window_count=last(windows),))
    for first_window in first(windows):ctx.windows_per_batch:last(windows)
        _dda_batch!(ctx, first_window; output_window=first_window - first(windows) + 1)
    end
    return outputs
end

function _dda_processes!(ctx, count)
    project = dirname(Base.active_project())
    workers = addprocs(count; exeflags=`--project=$project --threads=1 --startup-file=no`,
        env=["OPENBLAS_NUM_THREADS" => "1", "OMP_NUM_THREADS" => "1"])
    try
        # Do not use or remove any Distributed workers owned by the caller.
        outputs = _dda_outputs(ctx)
        empty_outputs = map(m -> m === nothing ? nothing : similar(m, size(m, 1), 0), outputs)
        input = merge(ctx, empty_outputs)
        @sync for (index, worker) in enumerate(workers)
            windows = (fld((index - 1) * ctx.window_count, count) + 1):fld(index * ctx.window_count, count)
            @async begin
                remotecall_fetch(Core.eval, worker, Main, :(using DelayDifferentialAnalysis))
                result = remotecall_fetch(_dda_process_chunk, worker, input, windows)
                for (matrix, block) in zip(outputs, result)
                    matrix === nothing || (matrix[:, windows] = block)
                end
            end
        end
    finally
        rmprocs(workers)
    end
    return nothing
end
