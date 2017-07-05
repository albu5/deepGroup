data_dir = './ActivityDataset';
testseq = [1     4     5     6     8     2     7    28    35    11    10    26];
anno_dir = fullfile(data_dir, 'Final_annotations_split1');
Theta = [0, 45, 90, 135, 180, 225, 270, 315]*pi/180;

trainX = [];
trainY = [];
testX = [];
testY = [];
trainMeta = [];
testMeta = [];

%%
for seqi = 1:44
    if seqi == 39, continue, end
    if any(testseq == seqi), continue, end
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    display(annofile)
    anno = load(annofile);
    anno = anno.anno.anno_data;
    t = 31;
    while(t < anno.nframe)
        group_label = anno.groups.grp_label(t,:);
        group_act = anno.groups.grp_act(t,:);
        group_ids = unique(group_label);
        group_ids = group_ids(group_ids>0);
        for id = group_ids
            feat_vec = [];
            persons =  anno.people;
            nped = 0;
            pedi = 0;
            for person = persons
                pedi = pedi + 1;
                if group_label(pedi) ~= id, continue, end
                bbs = person.bbs(t-9:t, :);
                poses = person.pose(t-9:t);
                poses_debug = poses;
                poses_debug(poses == 0) = 1;
                thetas = Theta(poses_debug);
                cosines = cos(thetas).*double(poses ~= 0);
                sines = sin(thetas).*double(poses ~= 0);
                actions = person.action(t-9:t);
                bbf = person.bbs(t-29:t, :);
                bbs(:,1) = bbs(:,1) + bbs(:,3)/2;
                bbs(:,2) = bbs(:,2) + bbs(:,4)/1;
                bbf(:,1) = bbf(:,1) + bbf(:,3)/2;
                bbf(:,2) = bbf(:,2) + bbf(:,4)/1;
                validt = sum(bbf, 2)>0;
                if sum(validt) < 10, continue, end
                tvec = 1:30;
                tvec = tvec(validt);
                bias = ones(size(tvec));
                predictor = [bias', tvec'];
                [Bx, ~, ~] = regress(bbf(validt,1), predictor);
                [By, ~, ~] = regress(bbf(validt,2), predictor);
                [Bw, ~, ~] = regress(bbf(validt,3), predictor);
                [Bh, ~, ~] = regress(bbf(validt,4), predictor);
                Bx = Bx(2) * ones(size(actions));
                By = By(2) * ones(size(actions));
                Bw = Bw(2) * ones(size(actions));
                Bh = Bh(2) * ones(size(actions));
                fvec = [bbs, cosines', sines', actions'-1, Bx', By', Bw', Bh'];
                if numel(feat_vec) == 0
                    feat_vec = fvec;
                    nped = 1;
                else
                    feat_vec = cat(2, feat_vec, fvec);
                    nped = nped + 1;
                end
            end
            if nped < 1
                display('skipped this group')
            elseif nped < 10
                npad = 10-nped;
                padsize = [10, npad*11];
                feat_vec = cat(2, feat_vec, zeros(padsize));
                if numel(trainX) == 0
                    trainX = reshape(feat_vec, [1, 10, 10*11]);
                else
                    trainX(end+1, :, :) = feat_vec;
                end
                trainMeta(end+1, :) = [seqi, t, id, group_act(id), pedi];
                trainY(end+1) = group_act(id);
            else
                feat_vec = feat_vec(:, 1:110);
                if numel(trainX) == 0
                    trainX = reshape(feat_vec, [1, 10, 10*11]);
                else
                    trainX(end+1, :, :) = feat_vec;
                end
                trainMeta(end+1, :) = [seqi, t, id, group_act(id), pedi];
                trainY(end+1) = group_act(id);
            end
        end
        t = t + 10;
    end
end

%%
for seqi = 1:44
    if ~any(testseq == seqi), continue, end
    if seqi == 39, continue, end
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    display(annofile)
    anno = load(annofile);
    anno = anno.anno.anno_data;
    t = 31;
    while(t < anno.nframe)
        group_label = anno.groups.grp_label(t,:);
        group_act = anno.groups.grp_act(t,:);
        group_ids = unique(group_label);
        group_ids = group_ids(group_ids>0);
        for id = group_ids
            feat_vec = [];
            persons =  anno.people;
            nped = 0;
            pedi = 0;
            pedi_vec = zeros(1,30);
            for person = persons
                pedi = pedi + 1;
                if group_label(pedi) ~= id, continue, end
                pedi_vec(pedi) = 1;
                bbs = person.bbs(t-9:t, :);
                poses = person.pose(t-9:t);
                poses_debug = poses;
                poses_debug(poses == 0) = 1;
                thetas = Theta(poses_debug);
                cosines = cos(thetas).*double(poses ~= 0);
                sines = sin(thetas).*double(poses ~= 0);
                actions = person.action(t-9:t);
                bbf = person.bbs(t-29:t, :);
                bbs(:,1) = bbs(:,1) + bbs(:,3)/2;
                bbs(:,2) = bbs(:,2) + bbs(:,4)/1;
                bbf(:,1) = bbf(:,1) + bbf(:,3)/2;
                bbf(:,2) = bbf(:,2) + bbf(:,4)/1;
                validt = sum(bbf, 2)>0;
                if sum(validt) < 10, continue, end
                tvec = 1:30;
                tvec = tvec(validt);
                bias = ones(size(tvec));
                predictor = [bias', tvec'];
                [Bx, ~, ~] = regress(bbf(validt,1), predictor);
                [By, ~, ~] = regress(bbf(validt,2), predictor);
                [Bw, ~, ~] = regress(bbf(validt,3), predictor);
                [Bh, ~, ~] = regress(bbf(validt,4), predictor);
                Bx = Bx(2) * ones(size(actions));
                By = By(2) * ones(size(actions));
                Bw = Bw(2) * ones(size(actions));
                Bh = Bh(2) * ones(size(actions));
                fvec = [bbs, cosines', sines', actions'-1, Bx', By', Bw', Bh'];
                if numel(feat_vec) == 0
                    feat_vec = fvec;
                    nped = 1;
                    
                else
                    feat_vec = cat(2, feat_vec, fvec);
                    nped = nped + 1;
                end
            end
            if nped < 1
                display('skipped this group')
            elseif nped < 10
                npad = 10-nped;
                padsize = [10, npad*11];
                feat_vec = cat(2, feat_vec, zeros(padsize));
                if numel(testX) == 0
                    testX = reshape(feat_vec, [1, 10, 10*11]);
                else
                    testX(end+1, :, :) = feat_vec;
                end
                testMeta(end+1, :) = [seqi, t, id, group_act(id), pedi_vec];
                testY(end+1) = group_act(id);
            else
                feat_vec = feat_vec(:, 1:110);
                if numel(testX) == 0
                    testX = reshape(feat_vec, [1, 10, 10*11]);
                else
                    testX(end+1, :, :) = feat_vec;
                end
                testMeta(end+1, :) = [seqi, t, id, group_act(id), pedi_vec];
                testY(end+1) = group_act(id);
            end
        end
        t = t + 10;
    end
end

%%
trainX = reshape(trainX, [size(trainX, 1), size(trainX, 2)*size(trainX, 3)]);
testX = reshape(testX, [size(testX, 1), size(testX, 2)*size(testX, 3)]);
trainY = trainY';
testY = testY';
outdir = './split1\group_activity_data_action';
mkdir(outdir)
% csvwrite(fullfile(outdir, 'trainX.csv'), trainX);
% csvwrite(fullfile(outdir, 'trainY.csv'), trainY);
% csvwrite(fullfile(outdir, 'testX.csv'), testX);
% csvwrite(fullfile(outdir, 'testY.csv'), testY);
% csvwrite(fullfile(outdir, 'trainMeta.csv'), trainMeta);
csvwrite(fullfile(outdir, 'testMeta.csv'), testMeta);

