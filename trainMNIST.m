% trainMNIST:
% 输入层：手写数字28*28个神经元
% 隐藏层：n1,n2,...,ni
% 输出层：10个神经元，代表输入的图片属于是哪个数字
% 网络：Out = Sigmoid(...Sigmoid(A2 * Sigmoid(A1*In)))
% 袁沅祥，2019-7
% MATLAB version >= MATLAB 7.0.0.19920 (R14)

clear;clc;
%% MNIST数据读取与保存.
train_file = 'data/train-images.idx3-ubyte';
label_file = 'data/train-labels.idx1-ubyte';
if ~exist(train_file)
    fprintf('请将"data/train-images-idx3-ubyte.gz"解压.\n');
end
if ~exist(label_file)
    fprintf('请将"data/train-labels-idx1-ubyte.gz"解压.\n');
end
[Train, Label] = loadMNIST(train_file, label_file);
if isempty(Train) || isempty(Label)
    return
end
test_file = 'test/t10k-images.idx3-ubyte';
test_label = 'test/t10k-labels.idx1-ubyte';
if ~exist(test_file)
    fprintf('请将"test/10k-images-idx3-ubyte.gz"解压.\n');
end
if ~exist(test_label)
    fprintf('请将"test/t10k-labels-idx1-ubyte.gz"解压.\n');
end
[Test, Tag] = loadMNIST(test_file, test_label, true);
if isempty(Test) || isempty(Tag)
    return
end
clear train_file label_file test_file test_label;

%% 训练神经网络.
DNN = TrainDNN(Train, Label, Test, Tag, [28, 14], 1e-2);
