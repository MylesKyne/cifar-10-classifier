load('cifar-10-data.mat'); %loads the dataset into the program

data = double(data);
StudentID = 38650452;
rng(StudentID,"twister");

figure;
for i = 1:4
    idx = randi([1, size(data, 1)]);
    subplot(1,4,i);
    imagesc(squeeze(data(idx,:,:,:)));
    title(label_names(labels(idx)));
end
saveas(gcf,'SampleImages.png');

ClassesNum = 10;
Classes = randperm(ClassesNum, 3);

selectData = [];
selectLabels = [];
for i = 1:length(Classes)
    idx = find(labels == Classes(i));
    selectData = cat(1, selectData, data(idx, :, :, :));
    selectLabels = [selectLabels; labels(idx)];
end

NumImages = size(selectData, 1);
split_index = randperm(NumImages, NumImages / 2);
training_index = split_index';
testing_index = setdiff(1:NumImages, split_index)';

TrainingData = reshape(selectData(training_index, :, :, :), [numel(training_index), 32*32*3]);
TestingData = reshape(selectData(testing_index, :, :, :), [numel(testing_index), 32*32*3]);
TrainingLabels = selectLabels(training_index);
TestingLabels = selectLabels(testing_index);


% Model Training - K-Nearest Neighbour
k = 2;
knnEuclidean = knn_classifier(TrainingData, TrainingLabels, TestingData, k, 'euclidean');
knnCosine = knn_classifier(TrainingData, TrainingLabels, TestingData, k, 'cosine');

disp(knnEuclidean(1:10));
disp(knnCosine(1:10));
disp(TestingLabels(1:10));


%MATLAB built in models
tic;
svmModel = fitcecoc(trainingData, trainingLabels);
svmPredictions = predict(svmModel, testingData);
svmTime = toc;

tic;
treeModel = fitctree(trainingData, trainingLabels);
treePredictions = predict(treeModel, testingData);
treeTime = toc;

%Evaluation
knnEuclideanAccuracy = mean(knnEuclidean == TestingLabels);
knnCosineAccuracy = mean(knnCosine == TestingLabels);
svmAccuracy = mean(svmPredictions == TestingLabels);
treeAccuracy = mean(treePredictions == TestingLabels);

%Confusion matrices
EuclideanCM = confusionmat(TestingLabels, knnEuclidean);
CosineCM = confusionmat(TestingLabels, knnCosine);
svmCM = confusionmat(TestingLabels, svmPredictions);
treeCM = confusionmat(TestingLabels, treePredictions);

save('cw1.mat', 'Classes', 'training_index', 'knnEuclideanAccuracy', 'knnCosineAccuracy', 'svmAccuracy', 'treeAccuracy', 'EuclideanCM', 'CosineCM', 'svmCM', 'treeCM', 'svmTime', 'treeTime');


%KNN implementation
function predictions = knn_classifier(TrainingData, TrainingLabels,TestingData, k, distance_metric)
    predictions = zeros(size(TestingData, 1), 1);
    for i = 1:size(TestingData, 1)
        distances = pdist2(TestingData(i, :), TrainingData, distance_metric);
        [~, sortedIndices] = sort(distances);
        nearestLabels = TrainingLabels(sortedIndices(1:k));
        predictions(i) = mode(nearestLabels);
    end
end