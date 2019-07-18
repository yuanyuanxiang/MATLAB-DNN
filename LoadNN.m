function DNN = LoadNN(file)
%% 从指定目录加载现有神经网络.
% file: DNN存放目录.
% 返回DNN: cell数组，依次存放A1, A2, A3, ...和 E, Loss.
% 袁沅祥，2019-7

if nargin == 0
    file = 'DNN_s*.mat';
end

dnn = dir(file);
if ~isempty(dnn)
    load(dnn(end).name);
    fprintf('Load DNN [%s] succeed.\n', dnn(end).name);
else
    DNN = cell(0);
    fprintf('Load Deep Neural Networks failed.\n');
end

end
