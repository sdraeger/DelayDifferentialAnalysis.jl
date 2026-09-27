function _solve_cpu(problem::RegressionProblem)::SolvedBlock
    isempty(problem.design) && return _nan_block(size(problem.design, 2))
    factorization = try
        svd(problem.design; full=false)
    catch
        return _nan_block(size(problem.design, 2))
    end
    sigma_max = isempty(factorization.S) ? 0.0 : maximum(factorization.S)
    tolerance = max(size(problem.design)...) * eps(Float64) * max(sigma_max, 1.0)
    projected = factorization.U' * problem.fit_target
    scaled = zeros(Float64, length(projected))
    keep = factorization.S .> tolerance
    scaled[keep] = projected[keep] ./ factorization.S[keep]
    coefficients = factorization.V * scaled
    residual = problem.residual_target - problem.design * coefficients
    rmse = sqrt(sum(abs2, residual) / size(problem.design, 1))
    return SolvedBlock(coefficients, rmse)
end

_solve_cpu(problems::Vector{RegressionProblem}) = SolvedBlock[_solve_cpu(problem) for problem in problems]

"""
Threaded variant used by the engine: each regression problem is independent,
so results are bit-identical to the sequential version regardless of thread
count.
"""
function _solve_cpu_threaded(
    problems::Vector{RegressionProblem}; num_cores::Integer=Threads.nthreads(),
)::Vector{SolvedBlock}
    n = length(problems)
    count = min(num_cores, Threads.nthreads(), n)
    count <= 1 && return _solve_cpu(problems)
    solutions = Vector{SolvedBlock}(undef, n)
    @sync for worker in 1:count
        Threads.@spawn for i in worker:count:n
            @inbounds solutions[i] = _solve_cpu(problems[i])
        end
    end
    return solutions
end
