function DNN = TrainDNN(Train, Label, Test, Tag, hidden, alpha)
% 在给定的数据集上训练神经网络.
% Train: 给定数据集，每一列代表一个input.
% Label: 数据集归类标签，每一列代表一个output.
% Test: 给定测试集，每一列代表一个input.
% Tag: 测试集归类标签，每一列代表一个output.
% hidden: vector，中间各层神经元个数.
% alpha: 初始学习率.
% DNN: cell数组，依次存放A1, A2, A3, ...和 E, Loss.
% 袁沅祥，2019-7

%% 各层神经元数量
sx = size(Train, 1);    %输入层神经元个数
n = hidden;             %隐藏层神经元个数
sy = size(Label, 1);    %输出层神经元个数
q = length(n) + 1;      %权重矩阵的个数

%% 初始值
if nargin < 4
    alpha = 1e-2; % 初始学习率
end
iter = 1000; % 单次最大迭代次数
[DNN, state] = TrainRecovery([sx, n, sy]);% 恢复训练
start = size(DNN{end}, 2); % 上一次迭代次数
if state
    fprintf('DNN:迭代[%g]次,精度%g.\n', start, DNN{end}(3, end));
    return
end
fprintf('从第[%g]步开始迭代.\n', start);
p = alpha * 0.99^start;
lr = p * 0.99.^(0:iter); % 学习率随迭代次数衰减

%profile on;
%profile clear;
%% 开始迭代
num = size(Train, 2);
% 第一行存放误差，第二、三行存放准确率
errs = zeros(3, iter);
count = 0; EarlyStopping = 3; %DNN早停条件
queue = cell(EarlyStopping+1, 1); %存放最近几次DNN网络
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
    queue = circshift(queue, 1);
    queue{1} = DNN;
    e = mean(sqrt(sum(total.*total)));
    s = Accuracy(DNN, Train, Label);
    t = Accuracy(DNN, Test, Tag);
    best = max(errs(3, 1:i)); % 前i-1次最好的结果
    errs(1, i) = e; errs(2, i) = s; errs(3, i) = t;
    if t <= best
        count = count + 1;
        if count == EarlyStopping
            DNN = queue{end};
            Loss = SaveResult(DNN, DNN{end}, errs, i-EarlyStopping, 1);
            return
        end
    else
        count = 0;
    end
    % 保存权重
    if t >= 0
        Loss = SaveResult(DNN, DNN{end}, errs, i, 10);
    end
    fprintf('%g err=%g lr=%g acc=%g %g use %gs\n',i+start,e,alpha,s,t,toc);
end
%profile viewer;
