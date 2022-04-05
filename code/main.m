clear all;
clc;
close all;

% This is the main script

% Add code folder to the search path
addpath(pwd);

cd ../data/


%%% Declaration and initialization of variables

% collection of image files, importing the CT images
imds = imageDatastore(pwd,'IncludeSubfolders',true,'LabelSource','foldernames');

iterationsMCCV = 100;
Accuracy = ones(iterationsMCCV,1);
Precision = ones(iterationsMCCV,1);
Specificity = ones(iterationsMCCV,1);
Sensitivity = ones(iterationsMCCV,1);

ConfMat = cell(iterationsMCCV,1);
FinalConfMatr = zeros(2); % Final confusion matrix
miniBatchSizeValue = 16;
MaxEpochsValue = 8;
InitialLearnRateValue = 1e-3;
LearnRateScheduleValue = 'piecewise';
LearnRateDropFactorValue = 0.2000;
LearnRateDropPeriodValue = 3;

TrainingDataPortion = 0.8;
Time4Training = ones(iterationsMCCV,1);
Time4Test = ones(iterationsMCCV,1);

% Augmentation operations to perform on the training images
% Data augmentation helps prevent the network from overfitting and
% memorizing the exact details of the training images
pixelRange = [-50 50];
scaleRange = [0.8 1.2];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ... % reflection in X
    'RandYReflection',true, ... % reflection in Y
    'RandRotation',[0 360], ... % rotation
    'RandXTranslation',pixelRange, ... % translation in X
    'RandYTranslation',pixelRange, ... % translation in Y
    'RandXScale',scaleRange, ... % scaling in X
    'RandYScale',scaleRange); % scaling in Y


%  Monte Carlo cross-validation of 10 iterations
for j = 1:iterationsMCCV
    disp(['Iteration ' num2str(j) ' out of ' num2str(iterationsMCCV)])
    %  Divide the data into training and validation sets.
    % We used 80% of the images for training and 20% for validation.
    [imdsTrain,imdsValidation] = splitEachLabel(imds,TrainingDataPortion,'randomized');


    % Load the pretrained ResNet-18 convolutional neural network
    net = resnet18;
    % Thanks to a transfer learning strategy, ResNet-18 identifies the
    % presence of hemorrhages on non-contrast CT images of the brain.
    % Transfer learning is a technique that consists of reusing a model
    % that has been previously trained for a specific task as a starting
    % point for another similar task. This strategy allows to retrain
    % models with a smaller amount of data (models that have already been
    % trained in large amounts of data are reused), something very common
    % in medicine, and it allows to significantly reduce training time.

    % ResNet-18 requires input images of size 224-by-224-by-3, where 3 is
    % the number of color channels
    inputSize = net.Layers(1).InputSize;
    % In that sense, since the images have different spatial resolution,
    % they were resized to 224-by-224-by-3 pixels to make them compatible
    % with the input size of the network.

    % To carry out the transfer learning, the last layers of the network
    % architecture were replaced: the final fully connected layer was
    % modified to contain the same number of nodes as number of classes (2
    % in this case), and a new classification layer was added whose output
    % is based on the probabilities calculated by the Softmax layer.

    % We extracted the layer graph of ResNet-18.
    lgraph = layerGraph(net);

    % Names of the two layers to replace of the layer graph
    [learnableLayer,classLayer] = findLayersToReplace(lgraph);

    % We replaced the fully connected layer with a new fully connected
    % layer with the number of outputs equal to the number of classes in
    % the data set: 2 in this case
    numClasses = numel(categories(imdsTrain.Labels));
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',1, ...
        'BiasLearnRateFactor',1);

    % Replacing fully connected layer in the layer graph
    lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

    % Replacing the classification layer with a new one without class labels.
    % trainNetwork automatically sets the output classes of the layer at training time.
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

    % Freezing the weight of the initial layers
    layers = lgraph.Layers;
    connections = lgraph.Connections;
    % sets the learning rates of all the parameters of the first layers to
    % zero to speed up network training and prevent overfitting to the new
    % data set
    layers(1:12) = freezeWeights(layers(1:12));
    % creates a new layer graph with the layers and reconnect all the
    % layers in the original order
    lgraph = createLgraphUsingConnections(layers,connections);


    % Datastore object performing image augmentation and resizing the
    % images to use for network training
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);

    % valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSizeValue);
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSizeValue, ...
        'MaxEpochs',MaxEpochsValue, ...
        'InitialLearnRate',InitialLearnRateValue, ...
        'LearnRateSchedule', LearnRateScheduleValue, ...
        'LearnRateDropFactor', LearnRateDropFactorValue, ...
        'LearnRateDropPeriod', LearnRateDropPeriodValue, ...
        'Shuffle','every-epoch', ... %'ValidationData',augimdsValidation, 'ValidationFrequency',valFrequency,
        'Verbose',false, ...
        'Plots','training-progress');

    % Retraining the ResNet-18 network
    % time for tranining
    tic;
    net = trainNetwork(augimdsTrain,lgraph,options);
    Time4Training(j) = toc;
    delete(findall(0)); % Close training progress plot

    % Resizing the validation images without performing further data augmentation
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

    % Classify Validation Images
    % cuantification of the time for validation
    tic;
    [YPred,probs] = classify(net,augimdsValidation);
    Time4Test(j) = toc;

    Accuracy(j) = mean(YPred == imdsValidation.Labels)*100;

    %     figure;
    %     cm = confusionchart(imdsValidation.Labels,YPred,...
    %         'RowSummary','row-normalized', ...
    %         'ColumnSummary','column-normalized');
    ConfMat{j} = confusionmat(imdsValidation.Labels,YPred);
    FinalConfMatr = FinalConfMatr + ConfMat{j,1};

    Precision(j) = ConfMat{j}(1,1)/sum(ConfMat{j}(:,1))*100;
    Sensitivity(j) = ConfMat{j}(1,1)/sum(ConfMat{j}(1,:))*100;
    Specificity(j) = ConfMat{j}(2,2)/sum(ConfMat{j}(2,:))*100;



end

% time table
TimeTable = table(Time4Training,Time4Test)

% Performance table
PerformanceTable = table(Accuracy,Specificity,Sensitivity,Precision)

% Bar Chart with Error Bars

% Errors for accuracy
pd = fitdist(Accuracy,'Normal');
ci = paramci(pd);
errhigh(1) = ci(2,1)- median(Accuracy);
errlow (1) = median(Accuracy)-ci(1,1);

% Errors for precision
pd = fitdist(Precision,'Normal');
ci = paramci(pd);
errhigh(2) = ci(2,1)- median(Precision);
errlow (2) = median(Precision)-ci(1,1);

% Errors for sensitivity
pd = fitdist(Sensitivity,'Normal');
ci = paramci(pd);
errhigh(3) = ci(2,1)- median(Sensitivity);
errlow (3) = median(Sensitivity)-ci(1,1);

% Errors for specificity
pd = fitdist(Specificity,'Normal');
ci = paramci(pd);
errhigh(4) = ci(2,1)- median(Specificity);
errlow (4) = median(Specificity)-ci(1,1);

x = 1:4;

figure,
h = bar(x,median(table2array(PerformanceTable)));
xlabel(['Accuracy', 'Precision', 'Sensitivity','Specificity']);
ylabel('Percentage');

hold on
er = errorbar(x,median(table2array(PerformanceTable)),errlow,errhigh);
er.Color = [0 0 0];
er.LineStyle = 'none';
grid on;
xlabel('Metrics');
ylabel('Percentage');
ylim([50 100])
hold off

cd ../results/
saveas(h,'errorbar','pdf');
save('results','TimeTable','PerformanceTable');



% Image classification visualization

img = imread(imdsValidation.Files{13});
img = imresize(img,inputSize(1:2));

[classfn,score] = classify(net,img);
% imshow(img);
% title(sprintf("%s (%.2f)", classfn, score(classfn)));

% Grad-CAM Reveals the Why Behind Deep Learning Decisions
scoreMap = gradCAM(net,img,classfn);
figure;
imshow(img);
hold on;
imagesc(scoreMap,'AlphaData',0.5);
colormap jet
hold off;
title("Grad-CAM");

%
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

