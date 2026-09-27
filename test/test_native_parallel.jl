using Distributed

function test_native_equal(actual, expected)
    @test actual.window_markers == expected.window_markers
    @test actual.channel_labels == expected.channel_labels
    @test [f.id for f in actual.flavors] == [f.id for f in expected.flavors]
    for (a, b) in zip(actual.flavors, expected.flavors)
        @test a.row_labels == b.row_labels
        @test isequal(a.matrix, b.matrix)
    end
end

@testset "Native CPU parallelism" begin
    # Uneven final batches, nonzero bounds, repeated channels, all five flavors.
    samples = repeat(native_parity_fixture(), 9, 1)
    parameters = (
        channels=[3, 1, 2, 1], flavors=["ST", "CT", "CD", "DE", "SY"],
        window_length=32, window_step=16, delays=[1, 2], model_terms=[1, 2, 4],
        derivative_points=3, order=3, nr_tau=2, start=7, stop=830,
        ct_channel_pairs=[(1, 3), (2, 3)], cd_channel_pairs=[(1, 2), (3, 2)],
        ct_window_length=2, ct_window_step=1, channel_labels=["a", "b", "c"],
    )
    original = copy(samples)
    expected = run_dda_matrix(samples; parameters..., num_cores=1)
    test_native_equal(run_dda_matrix(samples; parameters...), expected)
    for cores in (2, typemax(Int))
        test_native_equal(run_dda_matrix(samples; parameters..., num_cores=cores), expected)
    end
    test_native_equal(run_dda_matrix(samples; parameters..., num_cores=2, device="CPU"), expected)
    @test isequal(samples, original)

    single_parameters = merge(parameters, (start=0, stop=nothing))
    single = samples[1:39, :]
    single_expected = run_dda_matrix(single; single_parameters..., num_cores=1)
    @test length(single_expected.window_markers) == 1
    test_native_equal(run_dda_matrix(single; single_parameters..., num_cores=2), single_expected)

    @test_throws ArgumentError run_dda_matrix(samples; num_cores=0)
    @test_throws ArgumentError run_dda_matrix(samples; num_cores=-1)
    @test_throws ArgumentError run_dda_matrix(samples; parallelism=:invalid)
    @test_throws ArgumentError run_dda_matrix(samples; device="cuda:0", num_cores=2)
    @test_throws ArgumentError run_dda_matrix(samples; device="cuda", parallelism=:processes)

    @testset "Task-local scratch and one-term models" begin
        native = DelayDifferentialAnalysis.NativeDDA
        design = native._scratch_matrix(:design, (8, 1))
        target = native._scratch_matrix(:target, (8, 1))
        @test design !== target
        @test design === native._scratch_matrix(:design, (8, 1))
        other = fetch(Threads.@spawn native._scratch_matrix(:design, (8, 1)))
        @test design !== other
        prepared = native.PreparedWindow(reshape(collect(1.0:8.0), 8, 1),
            reshape(2.0 .* collect(1:8), 1, 8), 0)
        problem = native._group_problem(prepared, [1], [[0]], 8)
        @test problem.design[:, 1] == collect(1:8)
        @test native._solve_cpu(problem).coefficients ≈ [2.0]
        shifted = hcat(collect(1.0:8.0), collect(1.0:8.0).^2)
        derivative = permutedims(hcat(2 .* shifted[:, 1], -0.5 .* shifted[:, 2]))
        directed = native._directed_problem(native.PreparedWindow(shifted, derivative, 0),
            1, 2, 2, [[0]], 8)
        @test directed.design == shifted[:, [2, 1]]
        @test directed.fit_target == derivative[1, :]
        @test directed.residual_target == derivative[2, :]
        one_term = merge(parameters, (model_terms=[1],))
        test_native_equal(run_dda_matrix(samples; one_term..., num_cores=2),
            run_dda_matrix(samples; one_term..., num_cores=1))
    end

    @testset "Missing and rank-deficient windows" begin
        data = copy(samples)
        data[100:180, 2] .= NaN
        data[:, 3] .= 0.0
        test_native_equal(run_dda_matrix(data; parameters..., num_cores=2),
            run_dda_matrix(data; parameters..., num_cores=1))
    end

    @testset "Process parity and lifecycle" begin
        initial_workers = workers()
        owned_workers = addprocs(1; exeflags=`--threads=1 --startup-file=no`)
        try
            existing_workers = workers()
            test_native_equal(run_dda_matrix(samples; parameters..., num_cores=2,
                parallelism=:processes), expected)
            st_parameters = merge(parameters, (flavors=["ST"],))
            test_native_equal(run_dda_matrix(samples; st_parameters..., num_cores=2,
                parallelism=:processes), run_dda_matrix(samples; st_parameters...))
            @test workers() == existing_workers
            error_type = Sys.CPU_THREADS > 1 ? CompositeException : ErrorException
            @test_throws error_type run_dda_matrix(samples; parameters...,
                num_cores=2, parallelism=:processes, normalization="invalid")
            @test workers() == existing_workers
        finally
            rmprocs(owned_workers)
        end
        @test workers() == initial_workers
    end
end
