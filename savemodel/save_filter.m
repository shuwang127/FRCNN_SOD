function save_filter()
% -------------------------------------------------------------------------
% Save the caffemodel file into a format that you can use in Matlab.
% You should first add the caffe path in $(FRCN_ROOT)/external/caffe/*
% Written by Shu Wang.
% 29th Dec, 2015. 
% -------------------------------------------------------------------------

%% Clear all.
clear all;
clc;

%% Load the Caffe.Net and save in model file.
def = fullfile('..', 'models', 'VGG_CNN_M_1024_timely', 'test.prototxt');
net = fullfile('..', 'output', 'default', 'KakouTrain', 'vgg_cnn_m_1024_fast_rcnn_timely_iter_40000.caffemodel');
ConvNet = caffe.Net(def, net, 'test');
save model/ConvNet ConvNet

%% Save the common parameters.
NetPara.inputs = ConvNet.inputs;
NetPara.outputs = ConvNet.outputs;
NetPara.layer_names = ConvNet.layer_names;
NetPara.blob_names = ConvNet.blob_names;
NetPara.bottom_id_vecs = ConvNet.bottom_id_vecs;
NetPara.top_id_vecs = ConvNet.top_id_vecs;
save model/NetPara NetPara

%% Save the layer_vec parameters.
NetPara.layer_vec.conv1_weights = ConvNet.layers('conv1').params(1).get_data();
NetPara.layer_vec.conv1_biases = ConvNet.layers('conv1').params(2).get_data();
NetPara.layer_vec.conv2_weights = ConvNet.layers('conv2').params(1).get_data();
NetPara.layer_vec.conv2_biases = ConvNet.layers('conv2').params(2).get_data();
NetPara.layer_vec.conv3_weights = ConvNet.layers('conv3').params(1).get_data();
NetPara.layer_vec.conv3_biases = ConvNet.layers('conv3').params(2).get_data();
NetPara.layer_vec.conv4_weights = ConvNet.layers('conv4').params(1).get_data();
NetPara.layer_vec.conv4_biases = ConvNet.layers('conv4').params(2).get_data();
NetPara.layer_vec.conv5_weights = ConvNet.layers('conv5').params(1).get_data();
NetPara.layer_vec.conv5_biases = ConvNet.layers('conv5').params(2).get_data();
NetPara.layer_vec.fc6_weights = ConvNet.layers('fc6').params(1).get_data();
NetPara.layer_vec.fc6_biases = ConvNet.layers('fc6').params(2).get_data();
NetPara.layer_vec.fc7_weights = ConvNet.layers('fc7').params(1).get_data();
NetPara.layer_vec.fc7_biases = ConvNet.layers('fc7').params(2).get_data();
NetPara.layer_vec.cls_weights = ConvNet.layers('cls_score').params(1).get_data();
NetPara.layer_vec.cls_biases = ConvNet.layers('cls_score').params(2).get_data();
NetPara.layer_vec.bbox_weights = ConvNet.layers('bbox_pred').params(1).get_data();
NetPara.layer_vec.bbox_biases = ConvNet.layers('bbox_pred').params(2).get_data();
save model/NetPara NetPara

%% Save the blob_vec parameters.
NetPara.blob_vec.data = ConvNet.blobs('data').get_data();
NetPara.blob_vec.rois = ConvNet.blobs('rois').get_data();
NetPara.blob_vec.conv1 = ConvNet.blobs('conv1').get_data();
NetPara.blob_vec.norm1 = ConvNet.blobs('norm1').get_data();
NetPara.blob_vec.pool1 = ConvNet.blobs('pool1').get_data();
NetPara.blob_vec.conv2 = ConvNet.blobs('conv2').get_data();
NetPara.blob_vec.norm2 = ConvNet.blobs('norm2').get_data();
NetPara.blob_vec.pool2 = ConvNet.blobs('pool2').get_data();
NetPara.blob_vec.conv3 = ConvNet.blobs('conv3').get_data();
NetPara.blob_vec.conv4 = ConvNet.blobs('conv4').get_data();
NetPara.blob_vec.conv5 = ConvNet.blobs('conv5').get_data();
NetPara.blob_vec.pool5 = ConvNet.blobs('pool5').get_data();
NetPara.blob_vec.fc6 = ConvNet.blobs('fc6').get_data();
NetPara.blob_vec.fc7 = ConvNet.blobs('fc7').get_data();
NetPara.blob_vec.fc7_drop7_0_split_0 = ConvNet.blobs('fc7_drop7_0_split_0').get_data();
NetPara.blob_vec.fc7_drop7_0_split_1 = ConvNet.blobs('fc7_drop7_0_split_1').get_data();
NetPara.blob_vec.cls_score = ConvNet.blobs('cls_score').get_data();
NetPara.blob_vec.bbox_pred = ConvNet.blobs('bbox_pred').get_data();
NetPara.blob_vec.cls_prob = ConvNet.blobs('cls_prob').get_data();
save model/NetPara NetPara

%%
disp('done!');

end