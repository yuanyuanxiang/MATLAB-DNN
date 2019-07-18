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
[Train, Label] = loadMNIST(train_file, label_file);
if isempty(Train) || isempty(Label)
    return
end
test_file = 'test/t10k-images.idx3-ubyte';
test_label = 'test/t10k-labels.idx1-ubyte';
[Test, Tag] = loadMNIST(test_file, test_label, true);
if isempty(Test) || isempty(Tag)
    return
end

%% 各层神经元数量
sx = size(Train, 1);    %输入层神经元个数
n = [28,28,28];         %隐藏层神经元个数
sy = size(Label, 1);    %输出层神经元个数
q = length(n) + 1;      %权重矩阵的个数

%% 初始值
alpha = 1e-2; % 初始学习率
iter = 1000; % 单次最大迭代次数
DNN = TrainRecovery([sx, n, sy]);% 恢复训练
start = size(DNN{end}, 2); % 上一次迭代次数
fprintf('从第[%g]步开始迭代.\n', start);
p = alpha * 0.99^start;
lr = p * 0.99.^(0:iter); % 学习率随迭代次数衰减

%profile on;
%profile clear;
%% 开始迭代
num = size(Train, 2);
% 第一行存放误差，第二、三行存放准确率
errs = zeros(3, iter);
for i = 1:iter
    tic;
    alpha = lr(i);
    % 总误差
    total = zeros(sy, num);
    for k = 1 : num % 遍历元素
        % 前向传播
        X = cell(1, q+1);
        X{1} = Train(:, k); % input
        for p = 1:q
            X{p+1} = reLU(DNN{p} * [1; X{p}]);
        end
        err = X{q+1} - Label(:, k); % error
        total(:, k) = err;

        Store = DNN;
        % BP-误差反向传播
        for p = 1:q
            err = (DNN{end-p}(:, 2:end)' * err) .* Grad(X{q+2-p});
            Store{end-1-p} = DNN{end-1-p} - alpha * err * [1; X{q+1-p}]';
        end
        DNN = Store;
    end
    e = mean(sqrt(sum(total.*total)));
    s = Accuracy(DNN, Train, Label);
    t = Accuracy(DNN, Test, Tag);
    errs(1, i) = e; errs(2, i) = s; errs(3, i) = t;
    % 保存权重
    if t >= 0
        Loss = SaveResult(DNN, DNN{end}, errs, i, 10);
    end
    fprintf('%g err=%g lr=%g acc=%g %g use %gs\n',i+start,e,alpha,s,t,toc);
end
%profile viewer;
