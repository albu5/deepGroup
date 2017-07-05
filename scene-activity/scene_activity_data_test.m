data_dir = './ActivityDataset';
testseq =  [30    32    33    42    44    18    31    29    34    14    27];
anno_dir = fullfile(data_dir, 'Final_annotations');
outdir = './split4\group_activity_data_none';
Theta = [0, 45, 90, 135, 180, 225, 270, 315]*pi/180;
testMeta = csvread(fullfile(outdir, 'testMeta.csv'));
testResults = csvread(fullfile(outdir, 'testResults_none.csv'));

trainX1 = zeros(2000, 4);
trainX2 = zeros(2000, 20480);
trainY = zeros(2000, 1);
testX1 = zeros(2000, 4);
testX2 = zeros(2000, 20480);
testY = zeros(2000, 1);
testMetaScene = zeros(2000, 2);
train = 1;
test = 1;


%%
for seqi = 1:44
    if seqi == 39, continue, end
    if any(testseq == seqi), continue, end
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    display(annofile)
    anno = load(annofile);
    anno = anno.anno_data;
    t = 11;
    resnet50 = csvread(fullfile(data_dir, sprintf('seq%2.2d', seqi), 'resnet50.txt'));
    while(t < anno.nframe)
        group_label = anno.groups.grp_label(t,:);
        group_label = group_label(group_label>0);
        group_act = anno.groups.grp_act(t,:);
        
        card_vec = zeros(1,4);
        for id = group_label
            card_vec(group_act(id)) = card_vec(group_act(id)) + 1;
        end
        res_vec = resnet50(t-9:t,:);
        trainX1(train,:) = card_vec;
        trainX2(train,:) = reshape(res_vec, [1,2048*10]);
        trainY(train) = anno.Collective(t);
        t = t + 10;
        train = train + 1;
    end
end

%%
for seqi = 1:44
    if ~any(testseq == seqi), continue, end
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    display(annofile)
    anno = load(annofile);
    anno = anno.anno_data;
    t = 31;
    resnet50 = csvread(fullfile(data_dir, sprintf('seq%2.2d', seqi), 'resnet50.txt'));
    while(t < anno.nframe)
        group_label = anno.groups.grp_label(t,:);
        group_label = group_label(group_label>0);
        group_act = anno.groups.grp_act(t,:);
        
        idx = testMeta(:,1) == seqi & testMeta(:,2) == t;
        seqi, t, testResults(idx, :)
%         card_vec = zeros(1,4);
%         for i = 1:size(testResults, 1)
%             [~, blah] = max(testResults(i, :));
%             card_vec(blah) = card_vec(blah) + 1; 
%         end
        card_vec = (testMeta(idx, 5))' * (testResults(idx, :));
        if numel(testResults(idx, :)) == 0,
            t = t + 10;
            continue;
        else
%             card_vec = zeros(1,4);
%             for id = group_label
%                 card_vec(group_act(id)) = card_vec(group_act(id)) + 1;
%             end
            res_vec = resnet50(t-9:t,:);
            testX1(test, :) = card_vec;
            testX2(test, :) = reshape(res_vec, [1,2048*10]);
            testY(test) = anno.Collective(t);
            testMetaScene(test, :) = [seqi, t];
            test = test + 1;
            t = t + 10;
        end
    end
end

%%
trainX1(train:end, :) = [];
trainX2(train:end, :) = [];
trainY(train:end, :) = [];
testX1(test:end, :) = [];
testX2(test:end, :) = [];
testY(test:end, :) = [];
testMetaScene(test:end, :) = [];

%%
outdir = './split4\scene_activity_data_test_none';
mkdir(outdir)
% csvwrite(fullfile(outdir, 'trainX1.csv'), trainX1);
% csvwrite(fullfile(outdir, 'trainY.csv'), trainY);
% csvwrite(fullfile(outdir, 'testX1.csv'), testX1);
% csvwrite(fullfile(outdir, 'testY.csv'), testY);
% csvwrite(fullfile(outdir, 'trainX2.csv'), trainX2);
% csvwrite(fullfile(outdir, 'testX2.csv'), testX2);
csvwrite(fullfile(outdir, 'testMeta.csv'), testMetaScene);