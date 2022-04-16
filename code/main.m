% This is the main script of the paper “Intracerebral Hemorrhage Detection
% on Computed Tomography Images Using a Residual Neural Network”, please
% cite the paper and the code appropriately.
%
% Author: Miguel Altuve
% Email: miguelaltuve@gmail.com
% 2022


%% Clear Up
close all;
clear;
clc;


%% Adding code folder to the search path
% You must run main.m in the code directory
% cd C:\Users\miguel\ICHdetection\code
addpath(pwd);

cd ../data/ % data path


%% Declaration and initialization of variables
disp('Declaration and initialization of variables')

% Collection of image files, importing the CT images
imds = imageDatastore(pwd,'IncludeSubfolders',true,'LabelSource','foldernames');

% Initialization of the performance metrics
iterationsMCCV = 100; % Number of iterations of the MCCV
ConfMat = cell(iterationsMCCV,1); % confusion matrix of each iteration
FinalConfMatr = zeros(2); % Final confusion matrix
Accuracy = ones(iterationsMCCV,1); % Accuracy
Precision = ones(iterationsMCCV,1); % Precision
Specificity = ones(iterationsMCCV,1); % Specificity
Sensitivity = ones(iterationsMCCV,1); % Sensitivity

% Setting the hyperparameters of the model
miniBatchSizeValue = 16; % mini batch size
MaxEpochsValue = 8; % max epoch
InitialLearnRateValue = 1e-3; % Initial Learning Rate
LearnRateScheduleValue = 'piecewise'; % Learning Rate Schedule
LearnRateDropFactorValue = 0.2000; % Factor for dropping the learning rate
LearnRateDropPeriodValue = 3; % Learning Rate Drop Period

TrainingDataPortion = 0.8; % 80% of the data for training and 20% for validation
Time4Training = ones(iterationsMCCV,1); % Time during the training phase for each iteration
Time4Test = ones(iterationsMCCV,1); % Time during the validation phase for each iteration

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


%% Training and validating the models
disp('Training and validating the models')

% Monte Carlo cross-validation (MCCV) of iterationsMCCV iterations
for j = 1:iterationsMCCV
    disp(['Iteration ' num2str(j) ' out of ' num2str(iterationsMCCV)])

    %  Divide the data into training and validation sets.
    % We used 80% of the images for training and 20% for validation.
    [imdsTrain,imdsValidation] = splitEachLabel(imds,TrainingDataPortion,'randomized');

    % Loading the pretrained ResNet-18 network
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

    % Sets the learning rates of all the parameters of the first layers to
    % zero to speed up network training and prevent overfitting to the new
    % Data set
    layers(1:12) = freezeWeights(layers(1:12));

    % Creates a new layer graph with the layers and reconnect all the
    % layers in the original order
    lgraph = createLgraphUsingConnections(layers,connections);

    % Datastore object performing image augmentation and resizing the
    % images to use for network training
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);

    % Options for training deep learning neural network
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSizeValue, ...
        'MaxEpochs',MaxEpochsValue, ...
        'InitialLearnRate',InitialLearnRateValue, ...
        'LearnRateSchedule', LearnRateScheduleValue, ...
        'LearnRateDropFactor', LearnRateDropFactorValue, ...
        'LearnRateDropPeriod', LearnRateDropPeriodValue, ...
        'Shuffle','every-epoch', ...
        'Verbose',false, ...
        'Plots','training-progress');

    % Retraining the ResNet-18 network
    tic; % Computing the time for training
    net = trainNetwork(augimdsTrain,lgraph,options); % Tranining the network
    Time4Training(j) = toc;
    delete(findall(0)); % Closing training progress plot

    % Resizing the validation images without performing further data augmentation
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

    % Classification of new unseen images
    tic; % Computing the time for validation
    [YPred,scores] = classify(net,augimdsValidation);
    Time4Test(j) = toc;

    % Computing performance metris
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

end % End of MCCV

disp('Last confusion matrix on a validation subset')
disp(ConfMat{j})

disp('Sum of the confusion matrices on the validation subset over all iterations of the MCCV')
disp(FinalConfMatr)

disp('Confusion matrix averaged over all iterations of the MCCV')
disp(FinalConfMatr/iterationsMCCV)


%% Plotting and saving NumberImageToPlot random images with prediction labels
disp('Plotting four random images with prediction labels')

NumberImageToPlot = 20; 
idx = randperm(numel(imdsValidation.Files),NumberImageToPlot); % Random permutation

for i = 1:NumberImageToPlot
    figure
    cd ../data/ 
    I = readimage(imdsValidation,idx(i));
    imshow(I);
    Title = "Label = " + string(imdsValidation.Labels(idx(i))) + ...
        ", Prediction = " + string(YPred(idx(i))) + ...
        " (" + num2str(100*max(scores(idx(i),:)),3) + "%)";
    xlabel(Title);
    cd ../results/
    print(['ImagePrediction' num2str(i)],'-dpdf') % saving figure
end
close all;


%% Analyzing network decision making

% The gradient-weighted class activation mapping (Grad-CAM) technique was
% employed to visually explore and understand the network%s decisions

for i = 1:NumberImageToPlot
    cd ../data/ 
    I = readimage(imdsValidation,idx(i));
    img = imresize(I,inputSize(1:2)); % Resizing image

    % Classifying image using last trained network
    [YPred,scores] = classify(net,img); 

    % Grad-CAM reveals the Resnet decisions
    scoreMap = gradCAM(net,img,YPred);
    
    figure;
    imshow(img);
    hold on;
    imagesc(scoreMap,'AlphaData',0.5);
    colormap jet
    hold off;
    title("Grad-CAM");
    Title = "Label = " + string(imdsValidation.Labels(idx(i))) + ...
        ", Prediction = " + string(YPred) + ...
        " (" + num2str(100*max(scores),3) + "%)";
    xlabel(Title);

    cd ../results/
    print(['GradCAM' num2str(i)],'-dpdf')
end
close all;


%% Showing and saving the performance metrics of the MCCV
disp('Performance metrics of the MCCV');

% Time table
disp('Computation time during training and testing at each iteration of the MCCV')
TimeTable = table(Time4Training,Time4Test);
disp(TimeTable);

disp('Average time (s) during training and validation phases')
disp(mean(table2array(TimeTable)))

% Performance table
disp('Classification performance at each iteration of the MCCV')
PerformanceTable = table(Accuracy,Specificity,Sensitivity,Precision);
disp(PerformanceTable);

disp('Average of Accuracy, Specificity, Sensitivity, and Precision')
disp(mean(table2array(PerformanceTable)))

% saving performance metrics and the last ResNet model
save('results','TimeTable','PerformanceTable');
cd ../app
save('TrainedNetwork','net')


%% Visualization of the performance using error bars

% Error bars represent the median +- 95% confidence interval of the
% measures obtained on the validation set over the different iterations
% of the MCCV.

% Errors for accuracy
[errhigh(1), errlow(1)] = findErrorsLimits4ErrorBars(Accuracy);

% Errors for precision
[errhigh(2), errlow(2)] = findErrorsLimits4ErrorBars(Precision);

% Errors for sensitivity
[errhigh(3), errlow(3)] = findErrorsLimits4ErrorBars(Sensitivity);

% Errors for specificity
[errhigh(4), errlow(4)] = findErrorsLimits4ErrorBars(Specificity);

% Ploting bar chart with error bars
x = 1:4; % four bars
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
ylim([50 102])
hold off

cd ../results/
% saving error bars
print('errorbar','-dpdf'); 

% 
% 
% 
% That's All Folks