function genetic_optimization()
    % Define benchmark functions
    funcs = {@schwefel_102, @high_cond_elliptic_rot_func, @schwefel_102_noise_func};
    func_names = {"Schwefel 1.2", "High Conditioned Elliptic", "Schwefel 1.2 with Noise"};
    
    % Define optimization algorithms
    algorithms = {@genetic_algorithm};
    algo_names = {"Genetic Algorithm"};
    
    % Parameters
    dim = 10;
    lb = -100;
    ub = 100;
    num_runs = 15; % Number of independent runs
    max_iter = 500; % Increased max_iter
    pop_size = 50; % Increased pop_size
    
    % Run each algorithm on each function
    for i = 1:length(funcs)
        all_best_fitnesses = zeros(1, num_runs); % Store best fitnesses for each run
        all_best_solutions = zeros(num_runs, dim); % Store best solutions
        
        for run = 1:num_runs
            fprintf("Run %d: Running %s on function %d: %s\n", run, algo_names{1}, i, func_names{i});
            [best_sol, best_fitness] = algorithms{1}(funcs{i}, dim, lb, ub, max_iter, pop_size);
            all_best_fitnesses(run) = best_fitness;
            all_best_solutions(run, :) = best_sol;
        end
        
        % Calculate statistics
        avg_fitness = mean(all_best_fitnesses);
        std_fitness = std(all_best_fitnesses);
        [overall_best_fitness, best_run_index] = min(all_best_fitnesses);
        overall_best_solution = all_best_solutions(best_run_index, :);
        
        fprintf("\n--- Results for Function %d: %s ---\n", i, func_names{i});
        fprintf("Avg Fitness: %e\n", avg_fitness);
        fprintf("Std Dev Fitness: %e\n", std_fitness);
        fprintf("Best Fitness: %e\n", overall_best_fitness);
        fprintf("Best Solution: %s\n\n", num2str(overall_best_solution));
    end
 end
 
 %% Schwefel's Problem 1.2
 function f = schwefel_102(x)
    persistent o
    [ps, D] = size(x);
    if isempty(o)
        o = 1:D;  % Shift values: o_i = i
    end
    x = x - repmat(o, ps, 1);
    f = sum(cumsum(x, 2).^2, 2);
 end
 
 %% High Conditioned Elliptic Function
 function f = high_cond_elliptic_rot_func(x)
    persistent o M
    [ps, D] = size(x);
    if isempty(o)
        o = 1:D;  % Shift values: o_i = i
        
        % Generate rotation matrix M (Implementation from Appendix A)
        M = generate_rotation_matrix(D);
    end
    x = (x - repmat(o, ps, 1)) * M;
    a = 1e+6;
    f = sum(a.^((0:D-1)/(D-1)) .* x.^2, 2);
 end
 
 %% Schwefel's Problem 1.2 with Noise
 function f = schwefel_102_noise_func(x)
    persistent o
    [ps, D] = size(x);
    if isempty(o)
        o = 1:D;  % Shift values: o_i = i
    end
    x = x - repmat(o, ps, 1);
    f = sum(cumsum(x, 2).^2, 2) .* (1 + 0.4 * abs(randn(ps, 1)));  % Noise
 end
 
 %% Genetic Algorithm Optimization
 function [best_solution, best_fitness] = genetic_algorithm(func, dim, lb, ub, max_iter, pop_size)
    options = optimoptions('ga', ...
                           'PopulationSize', pop_size, ...
                           'MaxGenerations', max_iter, ...
                           'Display', 'off'); % Suppress display for cleaner output
    [best_solution, best_fitness] = ga(func, dim, [], [], [], [], lb, ub, [], options);
 end
 
 %  Rotation Matrix Generation (Implementation from Appendix A of CEC'05 Report)
 function M = generate_rotation_matrix(D)
    persistent cached_M;
    if isempty(cached_M)
        if D == 1
            cached_M = 1;
        else
            % Initialize
            M = eye(D);
            
            % Generate rotation matrix
            for i = 1:D-1
                for j = i+1:D
                    theta = pi * rand;
                    c = cos(theta);
                    s = sin(theta);
                    
                    R = eye(D);
                    R(i,i) = c;
                    R(j,j) = c;
                    R(i,j) = s;
                    R(j,i) = -s;
                    
                    M = M * R;
                end
            end
            cached_M = M;
        end
    end
    M = cached_M;
 end