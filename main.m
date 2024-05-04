% Step 1: Loading the Dataset
data = readtable('engine2_normalized.csv'); % Load the dataset (replace 'engine_data.csv' with your file path)

% Step 2: Data Preprocessing
data = fillmissing(data, 'nearest'); % Handle missing values (replace 'median' with 'mean' or 'nearest' if preferred)

% Step 3: Exploratory Data Analysis (EDA)
summary(data); % Display summary statistics
plot(data.engine_speed); % Plot a histogram of engine speed
xlabel('Engine Speed');
ylabel('Frequency');
title('Histogram of Engine Speed');

% Step 4: Feature Engineering
mean_engine_speed = mean(data.EngineSpeed); % Example: Extract mean engine speed
std_engine_speed = std(data.EngineSpeed); % Example: Extract standard deviation of engine speed

% Step 5: Model Development
rng(1); % Set random seed for reproducibility
idx = randperm(height(data));
train_idx = idx(1:round(0.8*length(idx)));
test_idx = idx(round(0.8*length(idx))+1:end);

train_data = data(train_idx,:);
test_data = data(test_idx,:);
