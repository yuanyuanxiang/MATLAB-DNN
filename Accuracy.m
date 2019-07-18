function s = Accuracy(DNN, Test, Tag)
%% 计算神经网络DNN在测试集Test上的准确率.Tag是测试集的标签.
% DNN: cell数组，依次存放A1, A2, A3, ...和 E, Loss.
% Test:数据集.
% Tag:数据集标签.
% s: DNN准确率.
% 袁沅祥，2019-7

if isempty(Test) || isempty(Tag)
    s = -1;
    return
end

n = length(DNN) - 2;
X = Test;
for i = 1:n
    X = reLU(DNN{i} * [ones(1, size(X, 2)); X]);
end
[a, p] = max(X);
[b, tag] = max(Tag);
s = mean(p == tag);

end
