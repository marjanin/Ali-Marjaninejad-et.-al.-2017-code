%% Linear_regression_model
% This code trains and tests the Linear Regression Model (LRM) as described
% in A. Marjaninejad et. al. 2017 and also defines the time periods in
% which the subject modes their finger

clear;close all;clc;                                                       % housekeeping
w_disp='on';                                                              % ploting on/off
load('sorted.mat')                                                         % loading the features on the workspace
load('b.mat')                                                              % loading the the descending list of the weights with the highest absolute values
end_time=7000;                                                             % the number of samples selected to rund the LRM on them with all the features. Increasing this number will also increase the execution time of the code. If it is too small, weights will not represent all different stages of movement and non movement
feat_num=200;                                                              % feat_num determines the number of the features with the highest absolute values of weights to be used in the model 

%% calculating the weights with the selected features
w=zeros(size(G_sorted,2),feat_num);                                        % predefining the variable w to speed up the code

for i=1:size(G_sorted,2)                                                   % looping over the number of fingers
    E_optim_sorted=E_sorted(:,b(i,1:feat_num));                            % only selecting a smaller set of weights with the highest absolute values of weights
    tic                                                                    % keeping the track of the execution time (tic)
    w(i,:)=pinv(E_optim_sorted(1:end_time,:)'*E_optim_sorted(1:end_time,:))*(E_optim_sorted(1:end_time,:)'*G_sorted(1:end_time,i));
    toc                                                                    % keeping the track of the execution time (toc)
end

%% estimate the finger movements using the LRM
estimated_output=zeros(size(G_sorted,2),size(E_sorted,1));                 % predefining the variable estimated_output to speed up the code
for i=1:size(G_sorted,2)                                                   % looping over the number of fingers
    estimated_output(i,:)=E_sorted(:,b(i,1:feat_num))*w(i,:)';             % estimating the ith finger movement
    if strcmp(w_disp,'on')                                                 % cheking the disp condition
        figure(i)                                                          % opening a new figure
        plot(G_sorted(:,i));hold on;plot(estimated_output(i,:),'r');       % ploting the estimated ith finger movement on the top of the real ith finger movement
        pause(0.1)                                                         % letting the figure to be disped before the execution of the rest of the code
    end
end

%% movement detection for the index finger
estimated_output_filtered=filter(ones(1,30)/30,1,estimated_output(2,:));   % smoothening the estimated output using a rectangular MA filter

if strcmp(w_disp,'on')                                                     % cheking the disp condition
    figure(2)                                                              % reopening figure 2
    plot(estimated_output_filtered,'g','linewidth',3);                     % ploting the smoothed estimated output plot on top of the other plots on figure 2
    pause(0.1)                                                             % letting the figure to be disped before the execution of the rest of the code
end

output_binary=zeros(size(estimated_output_filtered));                      % predefining the vector to represent the movement/non-movement periods
thresh=.4011*1.1;%0.6800;                                                  % setting an emperical threshold to detect finger movement

for i=1:length(estimated_output_filtered)                                  % looping over the filtered estimated output samples
    if (estimated_output_filtered(i)>thresh)                               % comparing the sample value with the threshold value
    output_binary(i)=5;                                                    % storing a non zero value (here 5) into the movement detection vector if the sample value is greater than the threshold value
    end
end

Cor1=corr2(G_sorted(end_time:end,1),estimated_output(1,end_time:end)')     % calculating the corrolation value of the real and estimated finger movements for the first finger
Cor2=corr2(G_sorted(end_time:end,2),estimated_output(2,end_time:end)')     % calculating the corrolation value of the real and estimated finger movements for the second finger
Cor3=corr2(G_sorted(end_time:end,3),estimated_output(3,end_time:end)')     % calculating the corrolation value of the real and estimated finger movements for the third finger
Cor4=corr2(G_sorted(end_time:end,4),estimated_output(4,end_time:end)')     % calculating the corrolation value of the real and estimated finger movements for the fourth finger
Cor5=corr2(G_sorted(end_time:end,5),estimated_output(5,end_time:end)')     % calculating the corrolation value of the real and estimated finger movements for the fifth finger

if strcmp(w_disp,'on')                                                     % cheking the disp condition
    figure(2)                                                              % making sure that the active figure is figure 2
    plot(output_binary,'m','linewidth',1)                                  % ploting the movement/non-movement signal on top of the other plots on figure 2
    axis([7001 9976 -2 7])                                                 % setting the axis
    xlabel('Smples')                                                       % setting the xlabel of the figure
    ylabel('Amplitude')                                                    % setting the ylabel of the figure
    title('Final results')                                                 % setting the title of the figure
    legend('Finger Flexion Signal','Linear Model Estimation','Smoothed Output','Finger Movement Detection') % setting the legend of the figure
    pause(0.1)                                                             % letting the figure to be disped before the execution of the rest of the code
end
%#ok<*NOPTS>                                                               % suppress editor message "Terminate statement with semicolon"                                                            

% Ali Marjaninejad - 2017
% If you are using the code, please cite: A. Marjaninejad, et. al. 2017
% Marjaninejad, Ali, Babak Taherian, and Francisco J. Valero-Cuevas. "Finger movements are mainly represented by a linear transformation of energy in band-specific ECoG signals." Engineering in Medicine and Biology Society (EMBC), 2017 39th Annual International Conference of the IEEE. IEEE, 2017.
