%% Fatemeh Esmaeili - 402816005 - Neuroscience Project
%% Long-term asynchronous decoding of arm motion using electrocorticographic signals in monkeys
%%
clear all;
clc;
close all;

%% Get a list of all .mat files in the current folder
matFiles = dir('*.mat');

% Initialize a variable to store the data
ECoG_Data = [];

% Read and collect data from all .mat files
for CH = 1:length(matFiles)
    % Load data from the .mat file
    filePath = matFiles(CH).name;
    data = load(filePath);
    
    % Assume the data is stored in a variable with a specific name
    % Replace 'variableName' with the actual variable name in your .mat files
    variableName = fieldnames(data);
    
    % Extract the number from the file name
    nameParts = regexp(matFiles(CH).name, '\d+', 'match');
    fileNumber = str2double(nameParts{1});

    % Store the data in the cell array at the position corresponding to the file number
    dataCell{fileNumber, :} = data.(variableName{1});
end

% Convert the cell array to a matrix (assuming data can be concatenated)
ECoG_Data = vertcat(dataCell{:});
clearvars -except ECoG_Data

% save('ECoG_Data.mat', 'ECoG_Data');

%% load ECoG_Data and Motion Data
load('ECoG_Data');
load('motion.mat');
Fs_ecog = 1000;
Fs_motion = 120;

%% step1: Pre_processng
%% Filter ECoG Data
% Design the Butterworth bandpass filter
[b,a] = butter(4,[0.1 600]/Fs_ecog);

% Apply the filter to each row of the dataMatrix
ECoG_Data_filt = filtfilt(b, a, double(ECoG_Data'))';

clearvars -except ECoG_Data ECoG_Data_filt Fs_ecog Fs_motion MotionData MotionTime

%% Remove Noisy Channel

% Calculate maximum amplitudes across rows (channels)
Max_Ampiltude = max(ECoG_Data_filt,[],2);

% Display a bar plot of maximum amplitudes
figure()
bar(Max_Ampiltude)

%  Remove pecific noisy channels
% ECoG_Data_filt([17],:)=[];
% ECoG_Data_filt([1,2,3,4,5,6,9,10,19,21],:)=[];

clearvars -except ECoG_Data ECoG_Data_filt Fs_ecog Fs_motion MotionData MotionTime

%% CAR

% Calculate the mean across channels for each time sample
meanAcrossChannels = mean(ECoG_Data_filt, 1);

% Subtract the mean from each channel's data
ECoG_Data_CAR = ECoG_Data_filt - meanAcrossChannels;

clearvars -except ECoG_Data ECoG_Data_filt Fs_ecog Fs_motion MotionData MotionTime ECoG_Data_CAR

%% Up-Sample Motion Position Signal

% Generate new time vector for up-sampling
MotionTime_new = linspace(MotionTime(1),MotionTime(end),size(ECoG_Data_CAR,2));

% Initialize cell array to store up-sampled motion position data
Positions{size(MotionData,1),1} = [];

% Up-sample each marker's motion data
for num_Marker = 1:size(MotionData,1)
    Positions{num_Marker,1} = interp1(MotionTime,MotionData{num_Marker,1},MotionTime_new);
end

clearvars -except ECoG_Data ECoG_Data_filt Fs_ecog Positions ECoG_Data_CAR

%% Step 2: Feature Extraction
%% Feature Extraction with Morlet Wavelet
down_sample_rate = 100; % Down-sampling rate
lag = 10; % Lag for feature extraction
FC = linspace(10,150,10); % Center frequencies for feature extraction
Feature = []; % Final matrix to store features

% Loop over each center frequency
for f = 1:length(FC)
    % Loop over each ECoG channel for feature extraction
    for CH = 1:size(ECoG_Data_CAR, 1)
        signal = ECoG_Data_CAR(CH, :); % Signal from the current channel
        
        % Parameters for Morlet wavelet transform
        fb = 0.5;
        n = 100;
        lb = -4;
        ub = 4;
        fc = FC(1, f);
        
        % Compute Morlet wavelet transform
        [psi, Input] = cmorwavf(lb, ub, n, fb, fc);
        
        % Apply Morlet transform to the signal
        w = conv(signal, real(psi));
        temp = abs(w);
        
        % Apply moving average and normalization
        temp = smooth(temp, 300, 'sgolay', 3);
        temp = (temp - mean(temp)) / std(temp);
        
        % Temporal down-sampling
        temp = temp(1:down_sample_rate:end);
        
        % Slice features using windowing
        feature_temp = [];
        for num_delay = 1:length(temp) - lag
            feature_temp(num_delay, :) = temp(num_delay:num_delay + lag - 1);
        end
        
        % Append extracted features to the final matrix
        Feature = [Feature, feature_temp];
    end
end

% Clear unnecessary variables
clearvars -except ECoG_Data ECoG_Data_filt Fs_ecog Positions ECoG_Data_CAR feature_final down_sample_rate lag
 
%% Train & Test Data

Input = Feature; % Input features for training and testing
Marker = 2;  % (e.g., 2.LSHO/3.LFLB/4.LWRI/5.RSHO/6.RELB/7.RWRI)
Pos = 3; % (1.X/2.Y/3.Z)
Output = Positions{Marker,1}(1:down_sample_rate:end,Pos);  % Output features for training and testing
% Remove initial rows to match input size
Output(1:lag-1,:) = [];

% Add bias term to input
Input = [ones(size(Input,1),1), Input];

% Split data into training and testing sets
num_train = floor(0.70 * length(Input)); % 70% of data for training
Input_train = Input(1:num_train,:);
Output_train = Output(1:num_train,:);
Input_test = Input(num_train+1:end,:);
Output_test = Output(num_train+1:end,:);

clearvars -except ECoG_Data ECoG_Data_filt Fs_eco Positions ECoG_Data_CAR feature_final down_sample_rate lag Input_train Output_train Input_test Output_test

%% Step 3: PLS Regresion
%% Choose Components with Cross-Validation (CV)

PLS_Comp = 100; % Number of PLS components to consider

[XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(Input_train, Output_train, PLS_Comp, 'cv', 10);
% XL: Predictor scores
% YL: Response scores
% BETA: coefficient estimates
% PCTVAR: percentage of variance  
% MSE: estimated mean squared errors  
% stats: PLS weights

% Plot cumulative percent variance explained in Output
figure()
plot(1:PLS_Comp, cumsum(100 * PCTVAR(2, :)), '-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in Y');

% Plot mean squared prediction error
figure()
plot(0:PLS_Comp, MSE(2, :), 'b-o');
xlabel('Number of components');
ylabel('Estimated Mean Squared Prediction Error');
legend({'PLSR'}, 'Location', 'Northeast');

%% Train PLS Model

clear XL YL XS YS BETA PCTVAR MSE stats Output_train_Predict Output_test_Predict
PLS_Comp = 11; % Number of PLS components to use

% Perform PLS regression with cross-validation
[XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(Input_train, Output_train, PLS_Comp, 'cv', 10);

% Predict outputs for training and testing data
Output_train_Predict = [ones(size(Input_train,1),1) Input_train] * BETA;
Output_test_Predict = [ones(size(Input_test,1),1) Input_test] * BETA;

% Plot actual versus predicted outputs for testing data
figure()
plot(Output_test)
hold on
plot(Output_test_Predict)
xlabel('Time')
ylabel('RWRI-X') % Change for each Marker and Possition
xlim([1 length(Output_test_Predict)])
legend('Observed trajectories', 'Predicted trajectories');
 
%% Evaluate Model Performance

% Calculate correlation coefficients between predicted and actual outputs
test_corr = corr2(Output_test, Output_test_Predict);
train_corr = corr2(Output_train, Output_train_Predict);

% Display correlation coefficients
disp(['Correlation for training data: ', num2str(train_corr), ', testing data: ', num2str(test_corr)]);

% Calculate R2 coefficient of determination, NRMS, and display results
R2 = var(Output_test_Predict) / var(Output_test);
NRMS = (sqrt(sum((Output_test - Output_test_Predict).^2) / length(Output_test_Predict))) / (max(Output_test) - min(Output_test));
disp(['R: ', num2str(test_corr), ', R2: ', num2str(R2), ', NRMS: ', num2str(NRMS)]);
