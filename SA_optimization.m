function SA_optimization()
    % Define benchmark functions
    funcs = {@schwefel_102, @high_cond_elliptic_rot_func, @schwefel_102_noise_func};
    func_names = {"Schwefel 1.2", "High Conditioned Elliptic", "Schwefel 1.2 with Noise"};
    
    % Define optimization algorithm
    algorithm = @simulated_annealing;
    algo_name = "Simulated Annealing";
    
    % Parameters
    dim = 10;
    lb = -100;
    ub = 100;
    num_runs = 15;
    max_iter = 5000; % Total iterations
    initial_temp = 1000;
    cooling_rate = 0.95;
    iterations_per_temp = 100; % Iterations at each temperature
    
    % Run SA on each function
    for i = 1:length(funcs)
        fprintf("\n--- Function %d: %s ---\n", i, func_names{i});
        
        all_best_fitnesses = zeros(1, num_runs);
        all_best_solutions = zeros(num_runs, dim);
        
        for run = 1:num_runs
            fprintf("  Run %d: Running %s\n", run, algo_name);
            
            % Call SA
            try
                if i == 2 % High Conditioned Elliptic
                    o = zeros(1, dim);
                    M = [];
                    try
                        warning('off', 'MATLAB:load:variableNotFound')
                        S = load('high_cond_elliptic_rot_data.mat', 'o', 'M');
                        warning('on', 'MATLAB:load:variableNotFound')
                        if isfield(S, 'o') && ~isempty(S.o)
                            o = S.o(1:min(length(S.o), dim));
                        end
                        if isfield(S, 'M') && ~isempty(S.M) && all(size(S.M) == [dim, dim])
                            M = S.M;
                        end
                        if ~exist('M', 'var') || isempty(M) || any(size(M) ~= [dim, dim]) || ~exist('o', 'var') || isempty(o)
                            M = generate_rotation_matrix(dim);
                            o = zeros(1, dim);
                        end
                    catch
                        warning('on', 'MATLAB:load:variableNotFound')
                        M = generate_rotation_matrix(dim);
                        o = zeros(1, dim);
                    end
                    [best_sol, best_fitness] = algorithm(funcs{i}, dim, lb, ub, max_iter, initial_temp, cooling_rate, iterations_per_temp, o, M);
                elseif i == 1 || i == 3 % Schwefel functions
                    o = 1:dim; % Shift for Schwefel 1.2
                    [best_sol, best_fitness] = algorithm(funcs{i}, dim, lb, ub, max_iter, initial_temp, cooling_rate, iterations_per_temp, o);
                else
                    [best_sol, best_fitness] = algorithm(funcs{i}, dim, lb, ub, max_iter, initial_temp, cooling_rate, iterations_per_temp);
                end
            catch ME
                fprintf('Error during optimization: %s\n', ME.message);
                continue;
            end
            
            all_best_fitnesses(run) = best_fitness;
            all_best_solutions(run, :) = best_sol;
        end
        
        % Calculate statistics
        avg_fitness = mean(all_best_fitnesses);
        std_fitness = std(all_best_fitnesses);
        [overall_best_fitness, best_run_index] = min(all_best_fitnesses);
        overall_best_solution = all_best_solutions(best_run_index, :);
        
        fprintf("\n  --- Results for %s ---\n", algo_name);
        fprintf("   Avg Fitness: %e\n", avg_fitness);
        fprintf("   Std Dev Fitness: %e\n", std_fitness);
        fprintf("   Best Fitness: %e\n", overall_best_fitness);
        fprintf("   Best Solution: %s\n\n", num2str(overall_best_solution));
    end
 end
 
 %% Simulated Annealing
 function [best_position, best_fitness] = simulated_annealing(func, dim, lb, ub, max_iter, initial_temp, cooling_rate, iterations_per_temp, varargin)
    
    current_position = lb + rand(1, dim) .* (ub - lb);
    best_position = current_position;
    best_fitness = func(current_position, varargin{:});
    current_fitness = best_fitness;
    
    temp = initial_temp;
    
    for iter = 1:max_iter
        for i = 1:iterations_per_temp
            % Generate a neighbor
            new_position = current_position + randn(1, dim); % Simple Gaussian neighborhood
            new_position = max(min(new_position, ub), lb); % Clip to bounds
            
            new_fitness = func(new_position, varargin{:});
            
            delta_fitness = new_fitness - current_fitness;
            
            % Acceptance probability
            if delta_fitness < 0 || rand() < exp(-delta_fitness / temp)
                current_position = new_position;
                current_fitness = new_fitness;
                
                if new_fitness < best_fitness
                    best_position = new_position;
                    best_fitness = new_fitness;
                end
            end
        end
        
        % Cool the temperature
        temp = temp * cooling_rate;
        
        % Optional: Termination condition (e.g., if temp is very low)
        if temp < 1e-6
            break;
        end
    end
 end
 
 %% (Functions and Rotation Matrix - same as before)
 
 %% Schwefel's Problem 1.2
 function f = schwefel_102(x, o)
    z = x - o;
    f = sum(cumsum(z, 2).^2, 2);
 end
 
 %% High Conditioned Elliptic Function
 function f = high_cond_elliptic_rot_func(x, o, M)
    z = (x - o) * M;
    a = 1e+6;
    f = sum(a.^((0:size(x, 2)-1)/(size(x, 2)-1)) .* z.^2, 2);
 end
 
 %% Schwefel's Problem 1.2 with Noise
 function f = schwefel_102_noise_func(x, o)
    z = x - o;
    f = sum(cumsum(z, 2).^2, 2) .* (1 + 0.4 * abs(randn(size(x, 1), 1)));
 end
 
 %% Rotation Matrix Generation (as per CEC'05)
 function M = generate_rotation_matrix(D)
    persistent cached_M;
    if isempty(cached_M)
        if D == 1
            cached_M = 1;
        else
            M = eye(D);
            for i = 1:D-1
                for j = i+1:D
                    theta = pi * rand;
                    c = cos(theta);
                    s = sin(theta);
                    R = eye(D);
                    R(i, i) = c;
                    R(j, j) = c;
                    R(i, j) = s;
                    R(j, i) = -s;
                    M = M * R;
                end
            end
            cached_M = M;
        end
    end
    M = cached_M;
 end