%% selecting_features
% This code provides a sorted list of the features with the largest
% absolute values of weights

clear;close all;clc;                                                       % housekeeping
load('sorted.mat')                                                         % loading the features on the workspace
%% creating the descending order of the weights with respect to their absolute values
end_time=7000;                                                             % the number of samples selected to rund the LRM on them with all the features. Increasing this number will also increase the execution time of the code. If it is too small, weights will not represent all different stages of movement and non movement
tic                                                                        % keeping the track of the execution time (tic)
w0=zeros(size(G_sorted,2),size(E_sorted,2));                               % predefining the variable w0 to speed up the code
b=zeros(size(G_sorted,2),size(E_sorted,2));                                % predefining the variable b to speed up the code

for i=1:size(G_sorted,2)                                                   % looping over the number of fingers
    w0(i,:)=pinv(E_sorted(1:end_time,:)'*E_sorted(1:end_time,:))*(E_sorted(1:end_time,:)'*G_sorted(1:end_time,i)); % creating the weights of the LRM 
    toc                                                                    % keeping the track of the execution time (toc)
    [a, b(i,:)]=sort(abs(w0(i,:)),'descend');                              % getting a descending list of the indices of the weights with the highest absolute values
    disp(['progress: ',num2str(100*i/5),' percent'])                       % disping the progress in percentages 
end

save('b.mat','b');                                                         % saving the output: the descending list of the weights with the highest absolute values

% Ali Marjaninejad - 2017
% If you are using the code, please cite: A. Marjaninejad, et. al. 2017
% Marjaninejad, Ali, Babak Taherian, and Francisco J. Valero-Cuevas. "Finger movements are mainly represented by a linear transformation of energy in band-specific ECoG signals." Engineering in Medicine and Biology Society (EMBC), 2017 39th Annual International Conference of the IEEE. IEEE, 2017.
