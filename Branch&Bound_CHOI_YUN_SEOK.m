% Knapsack problem solved using branch and bound algorithm in MATLAB

% Define weights and values of items with random numbers
weights = randi([1, 10], 1, 11)';
values = randi([10, 200], 1, 11)';

% Knapsack capacity
capacity = 10;

% Objective function coefficients (Change sign to convert maximization problem to minimization)
f = -values;

% Inequality constraints: A*x <= b
A = weights'; % Transpose of weights as each weight corresponds to one variable
b = capacity;

% Integer constraints
intcon = 1:length(weights); % All variables are integer

% Lower and upper bounds
lb = zeros(length(weights), 1); % Lower bounds for variables
ub = ones(length(weights), 1); % Upper bounds for variables (binary variables)

% Solve the integer linear programming problem
[x, fval] = intlinprog(f, intcon, A, b, [], [], lb, ub);

% Identify selected items
selected_items = find(x > 0.5);

% Display results
disp("Selected items:");
disp(selected_items); % Indices of selected items
disp("Total value: ");
disp(-fval); % Optimal objective function value (converted back to original maximization problem)
disp("Total weight: ");
disp(weights' * x); % Total weight of selected items
