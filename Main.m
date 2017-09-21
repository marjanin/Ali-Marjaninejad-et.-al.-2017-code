%% Main
% This code first runs the feature_extraction.m to extract the features and
% then selecting_features.m to have a list of the features which are going to
% be used in LRM. Finally this code runs linear_regression_model.m to
% create the Linear Regression Model (LRM) and estimate finger movements
% from the ECoG signal as well as detecting the finger movement time
% periods.
% This is supplementary to A. Marjaninejad et. al. 2017
% you can download the input signal data used in this study from http://www.bbci.de/competition/iv/#dataset4

feature_extraction
%pause()                                                                    % paused to make sure user will get the information of the previous step before proceeding to the next one
selecting_features
%pause()                                                                    % paused to make sure user will get the information of the previous step before proceeding to the next one
linear_regression_model

% Ali Marjaninejad - 2017
% If you are using the code, please cite: A. Marjaninejad, et. al. 2017
% Marjaninejad, Ali, Babak Taherian, and Francisco J. Valero-Cuevas. "Finger movements are mainly represented by a linear transformation of energy in band-specific ECoG signals." Engineering in Medicine and Biology Society (EMBC), 2017 39th Annual International Conference of the IEEE. IEEE, 2017.
