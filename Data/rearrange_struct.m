% Reads provided.mat into a 1D flat array
%[trial1/channel1/epoch1, trial1/channel1/epoch2, trial1/channel1/epoch3, trial1/channel2/epoch1, ... , trial75/channel3, epoch3]

% Load the data in
load("4_fv.mat", "fv");
fv.x;
% fv.x(1, 1, 1)
% fv.x(1, 1, 2)

% Contains all trials
data = [];

% For each trial
for i = 1:75
    % For each channel
    channel = [];
    for j = 1:40
        % For each epoch
        trial = [];
        for k = 1:3
            % Create a new array with [average1, average2, average3]
            trial = [trial fv.x(k, j, i)];
        end
        % Create a new array with [channel1, channel2, channel3, etc.] where channel1 contains averages for each of 3 epochs
        channel = [channel trial];
    end
    data = [data channel];
end

data
save('4_fv_rearranged.mat', "data");