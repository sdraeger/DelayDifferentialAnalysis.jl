const _CUDA_BATCH_SIZE = 512

_cuda_module()::Module = OptionalDeps.require(:CUDA)

function _parse_device(device::AbstractString)::Tuple{Symbol,Int}
    token = lowercase(strip(device))
    token == "cpu" && return (:cpu, 0)
    match_result = match(r"^cuda(?::([0-9]+))?$", token)
    match_result === nothing && error(
        "Unsupported device $device; expected \"cpu\", \"cuda\", or \"cuda:N\"",
    )
    index = match_result.captures[1] === nothing ? 0 : parse(Int, match_result.captures[1])
    return (:cuda, index)
end

function _solve_problems(
    problems::Vector{RegressionProblem},
    device::AbstractString;
    num_cores::Integer=Threads.nthreads(),
)::Vector{SolvedBlock}
    backend, device_index = _parse_device(device)
    backend == :cpu && return _solve_cpu_threaded(problems; num_cores)
    cuda = _cuda_module()
    cuda.functional() || error("CUDA is not functional on this system")
    cuda.device!(device_index)
    return Base.invokelatest(_solve_gpu, problems, cuda)
end

function _solve_gpu(problems::Vector{RegressionProblem}, cuda::Module)::Vector{SolvedBlock}
    solutions = SolvedBlock[_nan_block(size(problem.design, 2)) for problem in problems]
    groups = Dict{Tuple{Int,Int},Vector{Int}}()
    for (index, problem) in enumerate(problems)
        push!(get!(groups, size(problem.design), Int[]), index)
    end

    for ((rows, columns), indices) in groups
        if rows < columns
            for index in indices
                solutions[index] = _solve_cpu(problems[index])
            end
            continue
        end
        for first in 1:_CUDA_BATCH_SIZE:length(indices)
            batch = indices[first:min(first + _CUDA_BATCH_SIZE - 1, end)]
            _solve_gpu_batch!(solutions, problems, batch, rows, columns, cuda)
        end
    end
    return solutions
end

function _solve_gpu_batch!(
    solutions::Vector{SolvedBlock},
    problems::Vector{RegressionProblem},
    indices::Vector{Int},
    rows::Int,
    columns::Int,
    cuda::Module,
)
    batch_size = length(indices)
    designs = Array{Float64}(undef, rows, columns, batch_size)
    fit_targets = Array{Float64}(undef, rows, batch_size)
    residual_targets = Array{Float64}(undef, rows, batch_size)
    for (batch_index, problem_index) in enumerate(indices)
        problem = problems[problem_index]
        designs[:, :, batch_index] = problem.design
        fit_targets[:, batch_index] = problem.fit_target
        residual_targets[:, batch_index] = problem.residual_target
    end

    gpu_designs = cuda.CuArray(designs)
    solve_designs = copy(gpu_designs)
    gpu_fit_targets = cuda.CuArray(fit_targets)
    solve_targets = reshape(copy(gpu_fit_targets), rows, 1, batch_size)
    design_batch = [view(solve_designs, :, :, index) for index in 1:batch_size]
    target_batch = [view(solve_targets, :, :, index) for index in 1:batch_size]
    _, _, info = cuda.CUBLAS.gels_batched!('N', design_batch, target_batch)

    coefficients = dropdims(solve_targets[1:columns, :, :]; dims=2)
    predictions = dropdims(
        sum(gpu_designs .* reshape(coefficients, 1, columns, batch_size); dims=2);
        dims=2,
    )
    gpu_residual_targets = cuda.CuArray(residual_targets)
    errors = sqrt.(vec(mean(abs2, gpu_residual_targets - predictions; dims=1)))
    host_coefficients = Array(coefficients)
    host_errors = Array(errors)
    host_info = Array(info)

    for (batch_index, problem_index) in enumerate(indices)
        solutions[problem_index] = if iszero(host_info[batch_index])
            SolvedBlock(host_coefficients[:, batch_index], host_errors[batch_index])
        else
            _solve_cpu(problems[problem_index])
        end
    end
    return nothing
end
