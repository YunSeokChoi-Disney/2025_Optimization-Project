function knapsackIntegerProgram()
    % Set random seed for reproducibility
    rng(42);  % Fix the random number generator for consistent results

    % Parameters
    n = 10;               % Number of items
    capacity = 200;       % Maximum weight the knapsack can carry

    % Generate random weights and values for items
    weights = randi([1 50], 1, n);   % Random weights between 1 and 50
    values = randi([1 100], 1, n);   % Random values between 1 and 100

    % Ensure that the total weight exceeds the capacity (to make the problem meaningful)
    while sum(weights) <= capacity
        weights = randi([1 50], 1, n);  % Regenerate weights until their sum exceeds capacity
    end

    % Integer Linear Programming Formulation
    f = -values(:);       % Objective coefficients (negative because intlinprog minimizes)
    A = weights;          % Inequality constraint matrix (1 row: total weight)
    b = capacity;         % Right-hand side of inequality (maximum knapsack capacity)

    intcon = 1:n;         % Indices of integer variables (all decision variables must be binary)
    lb = zeros(n, 1);     % Lower bounds of decision variables (0)
    ub = ones(n, 1);      % Upper bounds of decision variables (1)

    % Set solver options (suppress output display)
    options = optimoptions('intlinprog', 'Display', 'off');

    % Solve the integer linear programming problem
    [x, fval, exitflag, output] = intlinprog(f, intcon, A, b, [], [], lb, ub, options);

    % Display results
    selected_items = find(x > 0.5);  % Find items selected (x close to 1)
    disp('Selected item indices:');
    disp(selected_items');          % Display selected indices as row vector

    disp('Total value:');
    disp(-fval);                    % Objective value (negated because we minimized -value)

    disp('Total weight:');
    disp(sum(weights(selected_items)));  % Sum the weights of selected items

    disp('Weights of items:');
    disp(weights);                 % Show all item weights

    disp('Values of items:');
    disp(values);                  % Show all item values
end
