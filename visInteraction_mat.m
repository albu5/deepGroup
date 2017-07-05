%%
% This script helps in visualizing pariwise interaction annotated using pairwise.m GUI


% Set this path to appropriate directory where collective activity dataset is stored
data_dir = './ActivityDataset';
anno_dir = fullfile(data_dir, 'annotations');
seq_str = 'seq%2.2d';
anno_str = 'anno%2.2d.mat';
int_str = 'int%2.2d.mat';
im_str = 'frame%4.4d.jpg';
interact_labels = {'no-interaction'...
    'approaching',...
    'leaving',...
    'passing-by',...
    'facing-each-other',...
    'walking-together',...
    'standing-in-a-row',...
    'standing-side-by-side',...
    'no-conclusion'
    };

%%
for i = 31:44
    im_dir = fullfile(data_dir, sprintf(seq_str, i));
    anno_file = fullfile(anno_dir, sprintf(anno_str, i));
    display(im_dir)
    display(anno_file)
    anno = load(anno_file);
    anno = anno.anno;
    interaction = load(fullfile(anno_dir, sprintf(int_str, i)));
    interaction = interaction.interaction;
    time_vec = 1:anno.nframe;
    n_inter = 0;
    for m = 1:numel(anno.people)
        for n = 1:numel(anno.people)
            if m == n
                continue
            else
                n_inter = n_inter + 1;
            end
            for t = time_vec
                ped_m = anno.people(m);
                ped_n = anno.people(n);
                if any(ped_n.time == t) && any(ped_m.time == t)
                    frame = imread(fullfile(im_dir, sprintf(im_str, t)));
                    imagesc(frame), axis image, axis off;
                    rectangle('Position', ped_m.sbbs(:,ped_m.time == t), 'LineWidth', 2, 'EdgeColor','b')
                    rectangle('Position', ped_n.sbbs(:,ped_n.time == t), 'LineWidth', 2, 'EdgeColor','r')
                    label = interaction(m, n, t);
                    title([interact_labels{label}, ' ', num2str(label)])
                    if label == 1 || label == 5 || label == 6 || label == 8 || label == 7
                        pause(0.001)
                    else
                        pause(0.005)
                    end
                end
            end
        end
    end
end