function [DNN, state] = TrainRecovery(n)
%% 恢复之前的结果，接着进行训练；或者加载现有神经网络.
% n:各层神经元个数，其中按顺序第一个元素为输入层神经元的个数,
% 最后一个元素为输出层神经元的个数，其余元素为隐藏层的神经元个数.
% DNN: cell数组，依次存放A1, A2, A3, ...和 E, Loss.
% 袁沅祥，2019-7

DNN = LoadNN();

if isempty(DNN)
    % 从头开始训练.
    h = length(n); % 网络层数
    DNN = cell(1, h+1);
    for i = 1:h-1
        % 第一列为偏置项.
        DNN{i} = rand(n(i+1), n(i) + 1) - 0.5;
    end
    % 倒数第2个元素为零列和单位阵的组合.
    DNN{h} = [zeros(n(h), 1), eye(n(h))];
end

disp('DNN infomation:'); disp(DNN);

for i = 1:length(n)
    fprintf('第[%g]层神经元个数: %g.\n', i, n(i));
end

%% 检测此神经网络是否已训练完成.
loss = DNN{end}(3, :);
best = max(loss);
count = 0;
state = false;
for i = length(loss)-2:length(loss)
    if 0 <= loss(i) && loss(i) <= best
        count = count + 1;
        if count == 3
            state = true;
        end
    else
        break
    end
end
