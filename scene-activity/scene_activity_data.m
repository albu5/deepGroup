data_dir = './ActivityDataset';
testseq = [30    32    33    42    44    18    31    29    34    14    27];
anno_dir = fullfile(data_dir, 'Final_annotations');
Theta = [0, 45, 90, 135, 180, 225, 270, 315]*pi/180;

trainX1 = [];
trainX2 = [];
trainY = [];
testX1 = [];
testX2 = [];
testY = [];

%%
for seqi = 1:44
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
        trainX1(end+1,:) = card_vec;
        trainX2(end+1,:) = reshape(res_vec, [1,2048*10]);
        trainY(end+1) = anno.Collective(t);
        t = t + 10;
        
    end
end

%%
for seqi = 1:44
    if ~any(testseq == seqi), continue, end
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
        testX1(end+1,:) = card_vec;
        testX2(end+1,:) = reshape(res_vec, [1,2048*10]);
        testY(end+1) = anno.Collective(t);
        t = t + 10;
        
    end
end

%%
outdir = './split4\scene_activity_data_train';
mkdir(outdir)
csvwrite(fullfile(outdir, 'trainX1.txt'), trainX1);
csvwrite(fullfile(outdir, 'trainY.txt'), trainY);
csvwrite(fullfile(outdir, 'testX1.txt'), testX1);
csvwrite(fullfile(outdir, 'testY.txt'), testY);
csvwrite(fullfile(outdir, 'trainX2.txt'), trainX2);
csvwrite(fullfile(outdir, 'testX2.txt'), testX2);