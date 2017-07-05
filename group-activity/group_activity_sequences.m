data_dir = './ActivityDataset';
testseq = [30    32    33    42    44    18    31    29    34    14    27];
anno_dir = fullfile(data_dir, 'Final_annotations');
Theta = [0, 45, 90, 135, 180, 225, 270, 315]*pi/180;

trainX = zeros(150000, 10, 110);
trainY = zeros(150000,1);
testX = zeros(150000, 10, 110);
testY = zeros(150000,1);
ntrain = 1;
ntest = 1;
nperms = 5;
%%
for seqi = 1:44
    if seqi == 39, continue, end
    if any(testseq == seqi), continue, end
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    display(annofile)
    anno = load(annofile);
    anno = anno.anno_data;
    t = 31;
    while(t < anno.nframe)
        group_label = anno.groups.grp_label(t,:);
        group_act = anno.groups.grp_act(t,:);
        group_ids = unique(group_label);
        group_ids = group_ids(group_ids>0);
        for id = group_ids
            feat_vec = [];
            persons =  anno.people(group_label == id);
            nped = 0;
            for person = persons
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
                %                 padvec = feat_vec(rem(1:npad, nped) + 1, :);
                %                 feat_vec = cat(2, feat_vec, padvec);
                feat_vec = cat(2, feat_vec, zeros(padsize));
                for permi = 1:nperms
                    permvec = randperm(10);
                    feat_vec_perm = reshape(feat_vec, [1, 10, 10*11]);
                    if numel(trainX) == 0
                        trainX(ntrain, :, :) = feat_vec_perm(permvec,:,:);
                        trainY(ntrain,1) = group_act(id);
                        ntrain = ntrain + 1;
                    else
                        trainX(ntrain, :, :) = reshape(feat_vec_perm, [1, 10, 10*11]);
                        trainY(ntrain,1) = group_act(id);
                        ntrain = ntrain + 1;
                    end
                end
            else
                feat_vec = feat_vec(:, 1:110);
                for permi = 1:nperms
                    permvec = randperm(10);
                    feat_vec_perm = reshape(feat_vec, [1, 10, 10*11]);
                    if numel(trainX) == 0
                        trainX(ntrain, :, :) = feat_vec_perm(permvec,:,:);
                        trainY(ntrain,1) = group_act(id);
                        ntrain = ntrain + 1;
                    else
                        trainX(ntrain, :, :) = reshape(feat_vec_perm, [1, 10, 10*11]);
                        trainY(ntrain,1) = group_act(id);
                        ntrain = ntrain + 1;
                    end
                end
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
    anno = anno.anno_data;
    t = 31;
    while(t < anno.nframe)
        group_label = anno.groups.grp_label(t,:);
        group_act = anno.groups.grp_act(t,:);
        group_ids = unique(group_label);
        group_ids = group_ids(group_ids>0);
        for id = group_ids
            feat_vec = [];
            persons =  anno.people(group_label == id);
            nped = 0;
            for person = persons
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
                %                 padvec = feat_vec(rem(1:npad, nped) + 1, :);
                %                 feat_vec = cat(2, feat_vec, padvec);
                feat_vec = cat(2, feat_vec, zeros(padsize));
                if numel(testX) == 0
                    testX(ntest, :, :) = reshape(feat_vec, [1, 10, 10*11]);
                else
                    testX(ntest, :, :) = feat_vec;
                end
                testY(ntest,1) = group_act(id);
                ntest = ntest + 1;
            else
                feat_vec = feat_vec(:, 1:110);
                if numel(testX) == 0
                    testX(ntest, :, :) = reshape(feat_vec, [1, 10, 10*11]);
                else
                    testX(ntest, :, :) = feat_vec;
                end
                testY(ntest,1) = group_act(id);
                ntest = ntest + 1;
            end
        end
        t = t + 10;
    end
end

%%
trainX(ntrain:end,:,:) = [];
trainY(ntrain:end) = [];

testX(ntest:end,:,:) = [];
testY(ntest:end) = [];

trainX = reshape(trainX, [size(trainX, 1), size(trainX, 2)*size(trainX, 3)]);
testX = reshape(testX, [size(testX, 1), size(testX, 2)*size(testX, 3)]);
trainY = trainY;
testY = testY;
outdir = './split4\group_activity_data_train';
mkdir(outdir)
csvwrite(fullfile(outdir, 'trainX.csv'), trainX);
csvwrite(fullfile(outdir, 'trainY.csv'), trainY);
csvwrite(fullfile(outdir, 'testX.csv'), testX);
csvwrite(fullfile(outdir, 'testY.csv'), testY);