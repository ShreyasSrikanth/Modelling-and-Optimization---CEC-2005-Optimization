function PSO_optimization()
    % Define benchmark functions
    funcs = {@schwefel_102, @high_cond_elliptic_rot_func, @schwefel_102_noise_func};
    func_names = {"Schwefel 1.2", "High Conditioned Elliptic", "Schwefel 1.2 with Noise"};
    
    % Define optimization algorithms
    algorithms = {@particle_swarm_optimization}; % Only PSO
    algo_names = {"Particle Swarm Optimization"};
    
    % Parameters
    dim = 10;
    lb = -100;
    ub = 100;
    num_runs = 15;
    max_iter = 500;
    pop_size = 50;  % For PSO
    
    % Run PSO on each function
    for i = 1:length(funcs)
        fprintf("\n--- Function %d: %s ---\n", i, func_names{i});
        
        all_best_fitnesses = zeros(1, num_runs);
        all_best_solutions = zeros(num_runs, dim);
        
        for run = 1:num_runs
            fprintf("  Run %d: Running %s\n", run, algo_names{1}); % Always PSO
            
            % Call PSO
            try
                if i == 2 % High Conditioned Elliptic
                    o = zeros(1, dim); % Initialize o to avoid potential errors
                    M = []; % Initialize M to avoid potential errors
                    try
                        warning('off', 'MATLAB:load:variableNotFound') % Suppress warning
                        S = load('high_cond_elliptic_rot_data.mat', 'o', 'M');
                        warning('on', 'MATLAB:load:variableNotFound')  % Turn warning back on
                        if isfield(S, 'o') && ~isempty(S.o)
                            o = S.o(1:min(length(S.o), dim)); % Truncate o if necessary
                        end
                        if isfield(S, 'M') && ~isempty(S.M) && all(size(S.M) == [dim, dim])
                            M = S.M;
                        end
                        if ~exist('M', 'var') || isempty(M) || any(size(M) ~= [dim, dim]) || ~exist('o', 'var') || isempty(o)
                            M = generate_rotation_matrix(dim);
                            o = zeros(1, dim); % Reset o
                        end
                    catch
                        warning('on', 'MATLAB:load:variableNotFound') % Ensure warning is on in catch
                        M = generate_rotation_matrix(dim);
                        o = zeros(1, dim); % Reset o
                    end
                    [best_sol, best_fitness] = algorithms{1}(funcs{i}, dim, lb, ub, max_iter, pop_size, o, M);
                elseif i == 1 || i == 3 % Schwefel functions
                    o = 1:dim; % Shift for Schwefel 1.2
                    [best_sol, best_fitness] = algorithms{1}(funcs{i}, dim, lb, ub, max_iter, pop_size, o);
                else
                    [best_sol, best_fitness] = algorithms{1}(funcs{i}, dim, lb, ub, max_iter, pop_size);
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
        
        fprintf("\n  --- Results for %s ---\n", algo_names{1}); % Always PSO
        fprintf("   Avg Fitness: %e\n", avg_fitness);
        fprintf("   Std Dev Fitness: %e\n", std_fitness);
        fprintf("   Best Fitness: %e\n", overall_best_fitness);
        fprintf("   Best Solution: %s\n\n", num2str(overall_best_solution));
    end
 end
 
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
 
 %% Particle Swarm Optimization
 function [best_position, best_fitness] = particle_swarm_optimization(func, dim, lb, ub, max_iter, pop_size, varargin)
    inertia_weight = 0.729844;
    cognitive_coeff = 1.49618;
    social_coeff = 1.49618;
    
    positions = lb + rand(pop_size, dim) .* (ub - lb);
    velocities = -1 + 2 * rand(pop_size, dim);
    personal_best_positions = positions;
    personal_best_fitnesses = inf(pop_size, 1);
    
    fitness_values = arrayfun(@(i) func(positions(i, :), varargin{:}), 1:pop_size)';
    [global_best_fitness, best_particle_index] = min(fitness_values);
    global_best_position = positions(best_particle_index, :);
    
    for iter = 1:max_iter
        fitness_values = arrayfun(@(i) func(positions(i, :), varargin{:}), 1:pop_size)';
        update_personal_best = fitness_values < personal_best_fitnesses;
        personal_best_fitnesses(update_personal_best) = fitness_values(update_personal_best);
        personal_best_positions(update_personal_best, :) = positions(update_personal_best, :);
        
        [current_global_best_fitness, current_best_particle_index] = min(personal_best_fitnesses);
        if current_global_best_fitness < global_best_fitness
            global_best_fitness = current_global_best_fitness;
            global_best_position = personal_best_positions(current_best_particle_index, :);
        end
        
        r1 = rand(size(positions));
        r2 = rand(size(positions));
        velocities = inertia_weight * velocities + cognitive_coeff * r1 .* (personal_best_positions - positions) + social_coeff * r2 .* (global_best_position - positions);
        positions = positions + velocities;
        positions = max(min(positions, ub), lb);
    end
    
    best_position = global_best_position;
    best_fitness = global_best_fitness;
 end
 
 %% Rotation Matrix Generation (as per CEC'05)
 function M = generate_rotation_matrix(D)
    persistent cached_M; % Store the matrix for reuse
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