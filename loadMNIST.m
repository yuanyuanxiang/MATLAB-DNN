function [Train, Label] = loadMNIST(train_file, label_file, force)
% MNIST数据读取与保存.
% train_file = 'data/train-images.idx3-ubyte';
% label_file = 'data/train-labels.idx1-ubyte';
% 返回时将矩阵转置，即矩阵的每一列是一个结果.
% 如果第3个参数输入force为true，则强制从原始文件读取数据.
% 该函数参考了互联网上现有代码，包括多个来源，未指明具体作者.

if nargin < 3
    force = false;
end

% MATLAB7.0
VERSION = datenum(version('-date'));
r2013a = datenum('Jan 01, 2013');

if ~exist('train-images.mat', 'file') || force
    FID = fopen(train_file, 'rb');
    if FID == -1
        Train = [];
        Label = [];
        fprintf('File [%s] does not exist.\n', train_file);
        return
    end
    magic = fread(FID, 1, 'int32', 0, 'ieee-be');
    if VERSION > r2013a
        assert(magic == 2051, ['Bad magic number in ', train_file, '']);
    end

    numImages = fread(FID, 1, 'int32', 0, 'ieee-be');
    numRows = fread(FID, 1, 'int32', 0, 'ieee-be');
    numCols = fread(FID, 1, 'int32', 0, 'ieee-be');

    Train = fread(FID, inf, 'unsigned char');
    Train = reshape(Train, numCols, numRows, numImages);
    Train = permute(Train,[2 1 3]);

    fclose(FID);
    % Reshape to #pixels x #examples
    Train = reshape(Train, size(Train, 1) * size(Train, 2), size(Train, 3));
    % Convert to double and rescale to [0,1]
    % https://blog.csdn.net/weixin_41503009/article/details/83420189
    Train = double(Train) * (0.999 / 255) + 0.001;
    if ~force
        save('train-images.mat', 'Train');
    end
else
    % 已存在mat格式文件，则直接加载
    load('train-images.mat');
end

if ~exist('train-labels.mat', 'file') || force
    FID = fopen(label_file,'r');
    if FID == -1
        Train = [];
        Label = [];
        fprintf('File [%s] does not exist.\n', label_file);
        return
    end
    magic=readint32(FID);
    NumberofImages=readint32(FID);
    if VERSION > r2013a
        assert(magic == 2049, ['Bad magic number in ', train_file, '']);
        assert(NumberofImages == size(Train, 2));
    end
    Label = zeros(NumberofImages,10);
    for i = 1:NumberofImages
        temp = fread(FID,1);
        Label(i,temp+1) = 1;
    end

    fclose(FID);
    Label = Label';
    Label(Label==0) = 0.001;
    if ~force
        save('train-labels.mat','Label');
    end
else
    % 已存在mat格式文件，则直接加载
    load('train-labels.mat');
end
end

function [getdata]=readint32(FID)

data = [];
for i = 1:4
    f=fread(FID,1);
    data = strcat(data,num2str(dec2base(f,2,8)));
end
getdata = bin2dec(data);

end
