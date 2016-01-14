% Fast R-CNN demo (in matlab).

%% clear all
clear all;
close all;
clc;

%% add path
[folder, name, ext] = fileparts(mfilename('fullpath'));
caffe_path = fullfile(folder, '..', 'caffe-fast-rcnn', 'matlab', 'caffe');
addpath(caffe_path);

%% initialize the model
use_gpu = true;
% You can try other models here:
def = fullfile(folder, '..', 'models', 'VGG_CNN_M_1024_timely', 'test.prototxt');
net = fullfile(folder, '..', 'output', 'default', 'KakouTrain', 'vgg_cnn_m_1024_fast_rcnn_timely_iter_40000.caffemodel');
init_key = caffe('init', def, net, 'test');
if exist('use_gpu', 'var') && ~use_gpu
    caffe('set_mode_cpu');
else
    caffe('set_mode_gpu');
end
model.init_key = init_key;
model.stride = 16;
model.pixel_means = reshape([102.9801, 115.9465, 122.7717], [1 1 3]);

%% load boxes and image
im_id = '000003';
cls_inds = 1;
cls_names = {'car'};
box_file = fullfile(folder, '..', 'data', 'demo', 'VideoFrame_Proposal', [im_id '_boxes.mat']);
% Boxes were saved with 0-based indexing
ld = load(box_file);
boxes = single(ld.boxes) + 1;
clear ld;
im_file = fullfile(folder, '..', 'data', 'demo', 'VideoFrame_Images', [im_id '.jpg']);
im = imread(im_file);
im_show = im;

%% check the init key
if model.init_key ~= caffe('get_init_key')
    error('You probably need call fast_rcnn_load_net() first.');
end

%% image_pyramid
% [im_batch, scales] = image_pyramid(im, model.pixel_means, false);
multiscale = false;
if ~multiscale
    SCALES = [600];
    MAX_SIZE = 1000;
else
    SCALES = [1200 864 688 576 480];
    MAX_SIZE = 2000;
end
num_levels = length(SCALES);

im = single(im);
% Convert to BGR
im = im(:, :, [3 2 1]);
% Subtract mean (mean of the image mean--one mean per channel)
im = bsxfun(@minus, im, model.pixel_means);

im_orig = im;
im_size = min([size(im_orig, 1) size(im_orig, 2)]);
im_size_big = max([size(im_orig, 1) size(im_orig, 2)]);
scale_factors = SCALES ./ im_size;

max_size = [0 0 0];
for i = 1:num_levels
    if round(im_size_big * scale_factors(i)) > MAX_SIZE
        scale_factors(i) = MAX_SIZE / im_size_big;
    end
    ims{i} = imresize(im_orig, scale_factors(i), 'bilinear', ...
        'antialiasing', false);
    max_size = max(cat(1, max_size, size(ims{i})), [], 1);
end

im_batch = zeros(max_size(2), max_size(1), 3, num_levels, 'single');
for i = 1:num_levels
    im = ims{i};
    im_sz = size(im);
    im_sz = im_sz(1:2);
    % Make width the fastest dimension (for caffe)
    im = permute(im, [2 1 3]);
    im_batch(1:im_sz(2), 1:im_sz(1), :, i) = im;
end
scales = scale_factors';

%% project_im_rois
% [feat_pyra_boxes, feat_pyra_levels] = project_im_rois(boxes, scales);
widths = boxes(:,3) - boxes(:,1) + 1;
heights = boxes(:,4) - boxes(:,2) + 1;

areas = widths .* heights;
scaled_areas = bsxfun(@times, areas, (scales.^2)');
diff_areas = abs(scaled_areas - (224 * 224));
[~, feat_pyra_levels] = min(diff_areas, [], 2);

feat_pyra_boxes = boxes - 1;
feat_pyra_boxes = bsxfun(@times, feat_pyra_boxes, scales(feat_pyra_levels));
feat_pyra_boxes = feat_pyra_boxes + 1;

%% make the input blobs
rois = cat(2, feat_pyra_levels, feat_pyra_boxes);
% Adjust to 0-based indexing and make roi info the fastest dimension
rois = rois - 1;
rois = permute(rois, [2 1]);

input_blobs = cell(2, 1);
input_blobs{1} = im_batch;
input_blobs{2} = rois;

%% caffe forward
th = tic();
blobs_out = caffe('forward', input_blobs);
fprintf('fwd: %.3fs\n', toc(th));

bbox_deltas = squeeze(blobs_out{1})';
probs = squeeze(blobs_out{2})';

%% bbox_pred & nms
num_classes = size(probs, 2);
dets = cell(num_classes - 1, 1);
NMS_THRESH = 0.3;
% class index 1 is __background__, so we don't return it
for j = 2:num_classes
    cls_probs = probs(:, j);
    cls_deltas = bbox_deltas(:, (1 + (j - 1) * 4):(j * 4));
    %------------------bbox_pred------------------%
    % pred_boxes = bbox_pred(boxes, cls_deltas);
    if isempty(boxes)
        pred_boxes = [];
        return;
    end
    
    Y = cls_deltas;
    
    % Read out predictions
    dst_ctr_x = Y(:, 1);
    dst_ctr_y = Y(:, 2);
    dst_scl_x = Y(:, 3);
    dst_scl_y = Y(:, 4);
    
    src_w = boxes(:, 3) - boxes(:, 1) + eps;
    src_h = boxes(:, 4) - boxes(:, 2) + eps;
    src_ctr_x = boxes(:, 1) + 0.5 * src_w;
    src_ctr_y = boxes(:, 2) + 0.5 * src_h;
    
    pred_ctr_x = (dst_ctr_x .* src_w) + src_ctr_x;
    pred_ctr_y = (dst_ctr_y .* src_h) + src_ctr_y;
    pred_w = exp(dst_scl_x) .* src_w;
    pred_h = exp(dst_scl_y) .* src_h;
    pred_boxes = [pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h];
    %------------------bbox_pred------------------%
    cls_dets = [pred_boxes cls_probs];
    %---------------------nms---------------------%
    % keep = nms(cls_dets, NMS_THRESH);
    if isempty(cls_dets)
        pick = [];
        return;
    end
    
    x1 = cls_dets(:,1);
    y1 = cls_dets(:,2);
    x2 = cls_dets(:,3);
    y2 = cls_dets(:,4);
    s = cls_dets(:,end);
    
    area = (x2-x1+1) .* (y2-y1+1);
    [vals, I] = sort(s);
    
    pick = s*0;
    counter = 1;
    while ~isempty(I)
        last = length(I);
        i = I(last);
        pick(counter) = i;
        counter = counter + 1;
        
        xx1 = max(x1(i), x1(I(1:last-1)));
        yy1 = max(y1(i), y1(I(1:last-1)));
        xx2 = min(x2(i), x2(I(1:last-1)));
        yy2 = min(y2(i), y2(I(1:last-1)));
        
        w = max(0.0, xx2-xx1+1);
        h = max(0.0, yy2-yy1+1);
        
        inter = w.*h;
        o = inter ./ (area(i) + area(I(1:last-1)) - inter);
        
        I = I(find(o<=NMS_THRESH));
    end
    
    pick = pick(1:(counter-1));
    %---------------------nms---------------------%
    cls_dets = cls_dets(pick, :);
    dets{j - 1} = cls_dets;
end

%% showboxes
THRESH = 0.8;
for j = 1:length(cls_inds)
    cls_ind = cls_inds(j);
    cls_name = cls_names{j};
    I = find(dets{cls_ind}(:, end) >= THRESH);
    %------------------showboxes------------------%
    % showboxes(im_show, dets{cls_ind}(I, :));
    boxes_dets = dets{cls_ind}(I, :);
    image(im_show);
    axis image;
    axis off;
    set(gcf, 'Color', 'white');
    
    if ~isempty(boxes_dets)
        x1 = boxes_dets(:, 1);
        y1 = boxes_dets(:, 2);
        x2 = boxes_dets(:, 3);
        y2 = boxes_dets(:, 4);
        c = 'r';
        s = '-';
        line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', c, 'linewidth', 2, 'linestyle', s);
        for i = 1:size(boxes_dets, 1)
            text(double(x1(i)), double(y1(i)) - 2, sprintf('%.3f', boxes_dets(i, end)), 'backgroundcolor', 'r', 'color', 'w');
        end
    end
    %------------------showboxes------------------%
    title(sprintf('%s detections with p(%s | box) >= %.3f', cls_name, cls_name, THRESH))
    fprintf('\n> Press any key to continue\n');
    % pause;
end
