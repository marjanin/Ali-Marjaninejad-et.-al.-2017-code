%% feature_extraction.m
% This code extracts features as described in A. Marjaninejad et. al. 2017
% and saves them in "sorted.mat" file

clear;close all;clc                                                        % housekeeping
w_disp='on';                                                              % ploting on/off

%% reading the signal
load('sub1_comp.mat')                                                      % loading the input signals
ECoG_sig_train=train_data';                                                % ECoG_sig_train: The first dimension (rows) is the electrode number and the second is the sample number
number_of_channels=size(ECoG_sig_train,1);                                 % finding the number of channels
number_of_fingers=size(train_dg,2);                                        % finding the number of dataglove recordings (fingers)
%% creating the filter bank
fs=1000;                                                                   % sampling rate of the recorded signals is 1 KHz
f0=1;f1=60;f2=100;f3=200;                                                  % cutoff frequencies for the filter bank                                  
filter_order=50;                                                           % the order of the filters

bpFilt1 = designfilt('bandpassfir', 'FilterOrder', filter_order, ...       % setting up the first filter     
'CutoffFrequency1', f0, 'CutoffFrequency2', f1,...
'SampleRate', fs);
b1=bpFilt1.Coefficients;
a1=1;

bpFilt2 = designfilt('bandpassfir', 'FilterOrder', filter_order, ...       % setting up the second filter
'CutoffFrequency1', f1, 'CutoffFrequency2', f2,...
'SampleRate', fs);
b2=bpFilt2.Coefficients;
a2=1;

bpFilt3 = designfilt('bandpassfir', 'FilterOrder', filter_order, ...       % setting up the third filter
'CutoffFrequency1', f2, 'CutoffFrequency2', f3,...
'SampleRate', fs);
b3=bpFilt3.Coefficients;
a3=1;

number_of_bands=3;                                                         % defining the number of bands which are used in the filter bank

if strcmp(w_disp,'on')                                                     % cheking the disp condition
    figure(1)                                                              % opening figure 1
    subplot(311);[h1,f1]=freqz(bpFilt1);plot(f1/pi,abs(h1));axis([0 1 0 1.2]);xlabel('Normalized frequency (\times\pi)');ylabel('Amplitude (dB)');title('Filter - First band');grid on; % Frequency response for the first filter
    subplot(312);[h2,f2]=freqz(bpFilt2);plot(f2/pi,abs(h2));axis([0 1 0 1.2]);xlabel('Normalized frequency (\times\pi)');ylabel('Amplitude (dB)');title('Filter - Second band');grid on; % Frequency response for the second filter
    subplot(313);[h3,f3]=freqz(bpFilt3);plot(f3/pi,abs(h3));axis([0 1 0 1.2]);xlabel('Normalized frequency (\times\pi)');ylabel('Amplitude (dB)');title('Filter - Third band');grid on; % Frequency response for the third filter
end

disp('Filter Bank is created!')                                            % command window notification: Filter Bank is created!
pause(1.2)                                                                 % making sure that the figure is shown and the user has enough time to see the notification before moving to the next step

%% applying the filter bank
ECoG_sig_bands_train=zeros(size(ECoG_sig_train,1),3,size(ECoG_sig_train,2)); % The first dimension is the electrode number, the second is the band number and the third is the sample number.

for i=1:size(ECoG_sig_train,1)
    clc;
    disp(['Preprocessing - percentage: ',num2str(100*(i/size(ECoG_sig_train,1))),'% ','(Electrode#: ',num2str(i),')']) % disping the progress of Preprocessing step in percentages
    ECoG_sig_bands_train(i,1,:) = filtfilt(b1,a1,ECoG_sig_train(i,:)); % band pass filter (Zero-phase forward and reverse digital IIR filtering) - first band
    ECoG_sig_bands_train(i,2,:) = filtfilt(b2,a2,ECoG_sig_train(i,:)); % band pass filter (Zero-phase forward and reverse digital IIR filtering) - second band
    ECoG_sig_bands_train(i,3,:) = filtfilt(b3,a3,ECoG_sig_train(i,:)); % band pass filter (Zero-phase forward and reverse digital IIR filtering) - third band
    % ECoG_sig_bands_train: (number of channels, number of bands, number of samples)
end

if strcmp(w_disp,'on')                                                  % cheking the disp condition
    channel_number_to_show=50;                                                    % the channel number we want to be shown on figure 2
    figure(2)                                                              % opening figure 2
    subplot(411)                                                           % opening a subplot with 4 rows and 1 column (first element)
    plot(reshape(ECoG_sig_bands_train(channel_number_to_show,1,:),size(ECoG_sig_bands_train,3),1));xlabel('Samples');ylabel('Amplitude');title('ECoG Signal - First band'); % ploting the filtered ECoG signal samples for the selected channel - first band
    subplot(412)                                                           % opening a subplot with 4 rows and 1 column (second element)
    plot(reshape(ECoG_sig_bands_train(channel_number_to_show,2,:),size(ECoG_sig_bands_train,3),1));xlabel('Samples');ylabel('Amplitude');title('ECoG Signal - Second band'); % ploting the filtered ECoG signal samples for the selected channel - second band
    subplot(413)                                                           % opening a subplot with 4 rows and 1 column (third element)
    plot(reshape(ECoG_sig_bands_train(channel_number_to_show,3,:),size(ECoG_sig_bands_train,3),1));xlabel('Samples');ylabel('Amplitude');title('ECoG Signal - Third band'); % ploting the filtered ECoG signal samples for the selected channel - third band
    subplot(414)                                                           % opening a subplot with 4 rows and 1 column (fourth element)
    plot(ECoG_sig_train(channel_number_to_show,:));xlabel('Samples');ylabel('Amplitude');title('Full band');  % ploting the not filtered ECoG signal samples for the selected channel
end

%% Amplitude Modulating (AM) of the Signal
pause(.1)                                                                  % letting the figure 2 to be disped before the execution of the rest of the code
clc                                                                        % cleaning the command window
disp('Extracting features...')                                             % command window notification: Extracting features...
ECoG_sig_bands_train_FeaVec=zeros(size(ECoG_sig_bands_train,1),3,(size(ECoG_sig_bands_train,3)/40)); % initializing the band specific ECoG feature matrix

for i=1:size(ECoG_sig_train,1)                                             % looping over the electrode number
    for j=1:size(ECoG_sig_bands_train,3)/40                                % looping over the sample bins (bins of the size 40)
        ECoG_sig_bands_train_FeaVec(i,1,j)=sum(ECoG_sig_bands_train(i,1,1+(j-1)*40:j*40).^2); % each feature bin is the summation of the power of 40 samples (the first frequency band)
        ECoG_sig_bands_train_FeaVec(i,2,j)=sum(ECoG_sig_bands_train(i,2,1+(j-1)*40:j*40).^2); % each feature bin is the summation of the power of 40 samples (the second frequency band)
        ECoG_sig_bands_train_FeaVec(i,3,j)=sum(ECoG_sig_bands_train(i,3,1+(j-1)*40:j*40).^2); % each feature bin is the summation of the power of 40 samples (the third frequency band)       
    end
end

if strcmp(w_disp,'on')                                                  % cheking the disp condition
    channel_number_to_show=50;                                             % the channel number we want to be shown on figure 2
    figure(3)                                                              % opening figure 3
    subplot(311)                                                           % opening a subplot with 3 rows and 1 column (first element)
    plot(reshape(ECoG_sig_bands_train_FeaVec(channel_number_to_show,1,:),size(ECoG_sig_bands_train_FeaVec,3),1));xlabel('Samples');ylabel('Amplitude');title('Features - First band'); % ploting the filtered ECoG signal samples for the selected channel - first band
    subplot(312)                                                           % opening a subplot with 3 rows and 1 column (second element)
    plot(reshape(ECoG_sig_bands_train_FeaVec(channel_number_to_show,2,:),size(ECoG_sig_bands_train_FeaVec,3),1));xlabel('Samples');ylabel('Amplitude');title('Features - Second band'); % ploting the filtered ECoG signal samples for the selected channel - second
    subplot(313)                                                           % opening a subplot with 3 rows and 1 column (second element)
    plot(reshape(ECoG_sig_bands_train_FeaVec(channel_number_to_show,3,:),size(ECoG_sig_bands_train_FeaVec,3),1));xlabel('Samples');ylabel('Amplitude');title('Features - Third band'); % ploting the filtered ECoG signal samples for the selected channel - third band
end

%% Post processing
pause(.1)                                                                  % letting the figure 3 to be disped before the execution of the rest of the code
clc                                                                        % cleaning the command window
disp('Decimating data...')                                                 % command window notification: Decimating data...
Glove_sig_train=zeros(number_of_fingers,length(decimate(train_dg(:,1),40))); % predefining the variable Glove_sig_train to speed up the code

for i=1:5                                                                  % looping over the number of fingers
    Glove_sig_train(i,:)=decimate(train_dg(:,i),40)';                      % decimating the data-glove data so that its size matched with the features' size
end

if strcmp(w_disp,'on')                                                     % cheking the disp condition
    figure(4)                                                              % opening figure 4
    finger_number_to_show=1;                                               % the finger number we want to be shown on figure 4
    plot(Glove_sig_train(finger_number_to_show,:))                         % plotting the decimated movement signal for the selected finger
    xlabel('Samples')
    ylabel('Amplitude')
    title('Thumb movement signal')
end
pause(.1)                                                                  % letting the figure 3 to be disped before the execution of the rest of the code
memory=25;                                                                 % defining the size of the memory to be fed as the input feature samples
E_fifea=zeros(number_of_channels,number_of_bands,size(ECoG_sig_bands_train_FeaVec,3)-memory+1,memory); % predefining the variable E_fifea to speed up the code

for j=1:number_of_channels                                                 % looping over the number of channels
    for b=1:number_of_bands                                                % looping over the number of frequency bands
        k=0;                                                               % a temporary counter
        for i=memory:size(ECoG_sig_bands_train_FeaVec,3)                   % loopin over the number of feature samples
            k=k+1;                                                         % increasing the temporary counter
            E_fifea(j,b,k,:)=ECoG_sig_bands_train_FeaVec(j,b,i-(memory-1):i); %% final features for the ECoG data (input) : packs of summed power samples of the current time and the past x values where x is defined by the "memory" variable
        end
    end
    clc;                                                                   % cleaning the command window
    disp(['Creating memory features - percentage: ',num2str(100.*j/number_of_channels),'%']) % disping the progress of "Creating memory features" step in percentages 
end

%% Combining all the features into a single database 
k=0;                                                                       % a temporary counter                        
E_fifea_sorted=zeros(number_of_channels*number_of_bands*memory,size(E_fifea,3)); % predefining the variable E_fifea_sorted to speed up the code

for i=1:number_of_channels                                                 % looping over the number of channels
    for j=1:number_of_bands                                                % looping over the number of bands
        for m=1:memory                                                     % looping over the memory
            k=k+1;                                                         % increasing the temporary counter
            E_fifea_sorted(k,:)=E_fifea(i,j,:,m);                          % flattening the memory, band, and channel number
        end
        clc                                                                % cleaning the command window
        disp(['Combining all the features into a single database - percentage: ',num2str(100*k/(number_of_channels*3*memory)),'%']) % disping the progress of "Combining all the features into a single database" step in percentages 
    end
end

E_sorted=E_fifea_sorted';                                                  % transposing the output file for ECoG features
G_sorted=Glove_sig_train(:,memory:end)';                                   % delaying samples by the number of samples defined by "memory" variable and transposing the finger movement features
save('sorted.mat','E_sorted','G_sorted')                                   % save the outputs to the sorted.mat file
disp('The execution of the feature_extraction.m is completed')             % command window notification: The execution of the code is completed

% Ali Marjaninejad - 2017
% If you are using the code, please cite: A. Marjaninejad, et. al. 2017
% Marjaninejad, Ali, Babak Taherian, and Francisco J. Valero-Cuevas. "Finger movements are mainly represented by a linear transformation of energy in band-specific ECoG signals." Engineering in Medicine and Biology Society (EMBC), 2017 39th Annual International Conference of the IEEE. IEEE, 2017.
