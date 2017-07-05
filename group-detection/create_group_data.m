% this script generates a struct array trainX, which stores all the group data
% used by kernel_loss_affinity.py
% Each entry of the struct array trainX contains features of individuals in a frame and ther group labels


% Set this path to appropriate directory where collective activity dataset is stored
data_dir = './ActivityDataset';

% directory where annotations with group data are saved
anno_dir = fullfile(data_dir, 'my_anno');

% exist(anno_dir, 'dir')
pre_anno = 'data_%2.2d';
pre_seq = 'seq%2.2d';
out_dir = fullfile(data_dir, 'group_data');

%%
mkdir(out_dir);
trainX = {};
trainY = {};
testX = {};
testY = {};
colorstrs = ['r', 'g', 'b', 'm', 'k', 'c'];
testseq = [1     4     5     6     8     2     7    28    35    11    10    26];
%%
for seqi = 1:44
    seqdir = fullfile(data_dir, sprintf(pre_anno, seqi));
    annofile = fullfile(anno_dir, sprintf(pre_anno, seqi));
    anno = load(annofile);
    anno = anno.anno_data;
    t = 31;
    group_lab = anno.groups.grp_label;
    group_act = anno.groups.grp_act;
    while t < anno.nframe
        curr_groups = group_lab(t,:);
        persons = [];
%         try
%             holf off
%         catch
%         end
%         pause
%         I = imread(fullfile(data_dir, sprintf(pre_seq, seqi),...
%             sprintf('frame%4.4d.jpg', t)));
%         imshow(I), hold on
        for i = 1:numel(curr_groups)
            if curr_groups(i)>0
                persons(end+1).bbs = anno.people(i).bbs(t-29:t, :);
                persons(end).bbs(:,1) = persons(end).bbs(:,1) + persons(end).bbs(:,3)/2;
                persons(end).bbs(:,2) = persons(end).bbs(:,2) + persons(end).bbs(:,4);
                
                persons(end).pose = anno.people(i).pose(t-29:t);
                persons(end).action = anno.people(i).action(t-29:t);
                persons(end).group = curr_groups(i);
                validt = sum(persons(end).bbs, 2) > 0;
                if sum(validt) < 10
                    persons(end) = [];
                else
                    temp1 = oneHot(mode(persons(end).pose(validt)), 8);
                    temp2 = oneHot(mode(persons(end).action(validt)), 2);
                    persons(end).bbs = persons(end).bbs(validt, :);
                    persons(end).pose = oneHot(persons(end).pose(validt)', 8);
                    persons(end).action = oneHot(persons(end).action(validt)', 2);
                    
                    bbf = persons(end).bbs;
                    bias = ones(size(bbf,1),1);
                    predictor = [bias, (1:numel(bias))'];
                    [B2,~,R2] = regress(bbf(:,1),predictor);
                    [B3,~,R3] = regress(bbf(:,2),predictor);
                    [B4,~,R4] = regress(bbf(:,3),predictor);
                    [B5,~,R5] = regress(bbf(:,4),predictor);
%                     meanbbs = mean(bbf,1);
                    meanbbs = bbf(end, :);
                    fvec = horzcat(B2(:)', B3(:)',B4(:)',B5(:)', ...
                        mean(R2), mean(R3), mean(R4), mean(R5), ...
                        temp1, temp2, meanbbs);                    
                    [tx, ty] = getTriVertices(meanbbs, (1:8)*(temp1'), 2);
%                     fill(tx, ty, colorstrs(rem(i,6)+1), 'faceAlpha', '0.3')
                    fvec2 = [B2(2), B3(2), B4(2), B5(2), temp1, temp2, meanbbs];
                    fvec3 = [B2(2), B3(2), B4(2), B5(2),...
                        (1:8)*(temp1'), (1:2)*(temp2'),...
                        meanbbs, ...
                        tx', ty'];
                    persons(end).features = fvec3;
                end
            end
        end
        if numel(persons)>1
            trainX{end+1, 1} = persons;
        end
        t = t + 10;
    end
end