clear

% Set this path to appropriate directory where collective activity dataset is stored
data_dir = './ActivityDataset';

% directory where resnet features of images of individuals are saved
% run write_resnet_to_tracks.py to generate these features
anno_dir = fullfile(data_dir, 'csvanno-posenet');

% exist(anno_dir, 'dir')
pre_anno = 'resnet_data%2.2d.txt';
out_dir = fullfile(data_dir, 'individual_data');

%%
train = 1;
test = 1;
mkdir(out_dir);
trainX = zeros(12000, 20560);
trainY = zeros(12000, 10);
testX = zeros(12000, 20560);
testY = zeros(12000, 10);
testMeta = zeros(12000, 3);
trainMeta = zeros(12000, 3);
colorstrs = ['r', 'g', 'b', 'm', 'k', 'c'];
testseq = [30    32    33    42    44    18    31    29    34    14    27];

%%
for seqi = 1:44
    if any(testseq == seqi), continue, end
    if seqi == 39, continue, end
    seqdir = fullfile(data_dir, sprintf(pre_anno, seqi));
    annofile = fullfile(anno_dir, sprintf(pre_anno, seqi));
    display(annofile)
    anno = csvread(annofile);
    nped = max(anno(:,2));
    for ped = 1:nped
        anno_ped = anno(anno(:,2) == ped, :);
        i = 1;
        while i < size(anno_ped, 1)
            if i + 9 > size(anno_ped, 1)
                data = anno_ped(i:end, :);
                padlen = 10 - size(data, 1);
                padsize = size(data, 2);
                data =  vertcat(data, zeros(padlen, padsize));
            else
                data = anno_ped(i:i+9, :);
                padlen = 0;
            end
            if i < 20
                endi = min(i+9, size(anno_ped, 1));
                starti = max(1, i-20);
                bbf = anno_ped(starti:endi, 3:6);
            else
                endi = min(i+9, size(anno_ped, 1));
                starti = max(1, i-20);
                bbf = anno_ped(starti:endi, 3:6);
            end
            bias = ones(size(bbf,1),1);
            predictor = [bias, (1:numel(bias))'];
            [B2,~,R2] = regress(bbf(:,1),predictor);
            [B3,~,R3] = regress(bbf(:,2),predictor);
            [B4,~,R4] = regress(bbf(:,3),predictor);
            [B5,~,R5] = regress(bbf(:,4),predictor);
%             feat = data(:, [3:6, 11:2058]);
            e10 = ones(10, 1);
            if padlen > 0
                e10(end-padlen+1:end) = 0;
            end
            feat = horzcat(data(:, 3:6),...
                B2(2)*e10, B3(2)*e10, B4(2)*e10, B5(2)*e10,...
                data(:, 11:2058));
            label = data(:, 8);
            trainX(train, :) = single(reshape(feat, [1, 10*2056]));
            trainY(train, :) = reshape(label, [1, 10]);
            trainMeta(train, :) = [seqi, anno_ped(endi, 1:2)];
            train = train + 1;
            i = i + 10;
        end
    end
    debug = whos('trainX');
    display(train)
    display(debug.bytes/(2^20))
end

%%
for seqi = 1:44
    if ~any(testseq == seqi), continue, end
    if seqi == 39, continue, end
    seqdir = fullfile(data_dir, sprintf(pre_anno, seqi));
    annofile = fullfile(anno_dir, sprintf(pre_anno, seqi));
    display(annofile)
    anno = csvread(annofile);
    nped = max(anno(:,2));
    for ped = 1:nped
        anno_ped = anno(anno(:,2) == ped, :);
        i = 1;
        while i < size(anno_ped, 1)
            if i + 9 > size(anno_ped, 1)
                data = anno_ped(i:end, :);
                padlen = 10 - size(data, 1);
                padsize = size(data, 2);
                data =  vertcat(data, zeros(padlen, padsize));
            else
                data = anno_ped(i:i+9, :);
                padlen = 0;
            end
            if i < 20
                endi = min(i+9, size(anno_ped, 1));
                starti = max(1, i-20);
                bbf = anno_ped(starti:endi, 3:6);
            else
                endi = min(i+9, size(anno_ped, 1));
                starti = max(1, i-20);
                bbf = anno_ped(starti:endi, 3:6);
            end
            bias = ones(size(bbf,1),1);
            predictor = [bias, (1:numel(bias))'];
            [B2,~,R2] = regress(bbf(:,1),predictor);
            [B3,~,R3] = regress(bbf(:,2),predictor);
            [B4,~,R4] = regress(bbf(:,3),predictor);
            [B5,~,R5] = regress(bbf(:,4),predictor);
%             feat = data(:, [3:6, 11:2058]);
            e10 = ones(10, 1);
            if padlen > 0
                e10(end-padlen+1:end) = 0;
            end
            feat = horzcat(data(:, 3:6),...
                B2(2)*e10, B3(2)*e10, B4(2)*e10, B5(2)*e10,...
                data(:, 11:2058));
            label = data(:, 8);
            testX(test, :) = single(reshape(feat, [1, 10*2056]));
            testY(test, :) = reshape(label, [1, 10]);
            testMeta(test, :) = [seqi, anno_ped(endi, 1:2)];
            test = test + 1;
            i = i + 10;
        end
    end
    debug = whos('testX');
    display(test)
    display(debug.bytes/(2^20))
end

%%
trainX(train:end, :) = [];
trainY(train:end, :) = [];
testX(test:end, :) = [];
testY(test:end, :) = [];
testMeta(test:end, :) = [];
trainMeta(train:end, :) = [];


%% Save training and testing files
hdf5write('person_actions_posenet_meta_split4.h5', '/trainX', trainX);
hdf5write('person_actions_posenet_meta_split4.h5', '/trainY', trainY, 'WriteMode', 'append');
hdf5write('person_actions_posenet_meta_split4.h5', '/testX', testX, 'WriteMode', 'append');
hdf5write('person_actions_posenet_meta_split4.h5', '/testY', testY, 'WriteMode', 'append');
hdf5write('person_actions_posenet_meta_split4.h5', '/trainMeta', trainMeta, 'WriteMode', 'append');
hdf5write('person_actions_posenet_meta_split4.h5', '/testMeta', testMeta, 'WriteMode', 'append');
