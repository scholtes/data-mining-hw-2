% Garrett Scholtes
% 2016-09-24
% Homework #2
% Intelligent data analysis
% 
% NOTE: Created with MATLAB R2013a
%       Using `ClassificationTree.fit` in lieu of `fitctree`

clear all;
close all;
% Close tree view windows
hiddenfigs = findall(0,'Type','figure', '-not', 'HandleVisibility', 'on');
close(hiddenfigs);

% Seed the random number generator for convenience
% Can be removed if desired 
rng(2016);

% Part 1
% Load data + covariance matrix
d1 = xlsread('data_banknote_authentication.xlsx');

cm = cov(d1);

% Part 2
% Scatter plot attr-1 attr-2
D10 = d1(d1(:,5)==0,:);
D11 = d1(d1(:,5)==1,:);
scatter(D10(:,1),D10(:,2),'r.');
hold on;
scatter(D11(:,1),D11(:,2),'b.');
title('attr-2 vs attr-1');
xlabel('attr-1');
ylabel('attr-2');
legend('class 0', 'class 1');

% Part 3
% Scatter plot attr-1 attr-2
figure;
D20 = d1(d1(:,5)==0,:);
D21 = d1(d1(:,5)==1,:);
scatter(D20(:,2),D20(:,4),'m.');
hold on;
scatter(D21(:,2),D21(:,4),'c.');
title('attr-4 vs attr-2');
xlabel('attr-2');
ylabel('attr-4');
legend('class 0', 'class 1');

% Part 4
% Scatter plot attr-3 class

figure;
D30 = d1(d1(:,5)==0,:);
D31 = d1(d1(:,5)==1,:);
scatter(D30(:,3),D30(:,5),'k.');
hold on;
scatter(D31(:,3),D31(:,5),'r.');
title('class vs attr-3');
xlabel('attr-3');
ylabel('class');

% Part 5
% Scatter plot attr-2 class

figure;
D40 = d1(d1(:,5)==0,:);
D41 = d1(d1(:,5)==1,:);
scatter(D40(:,2),D40(:,5),'k.');
hold on;
scatter(D41(:,2),D41(:,5),'r.');
title('class vs attr-2');
xlabel('attr-2');
ylabel('class');

% close all;
% Part 6
% Select a training set of 1000 randomly selected records

% (1) Shuffle the index list [1, 2, 3, ... , size(d1)] 
permutation = randperm(size(d1, 1));
% (2) The first 1000 entries of those shuffles produces 
%     a random selection of indecies for the training set.
%     The last entries of the shuffle represent the test set.
TRAIN_SIZE = 1000;
train_set = d1(permutation(:,1:TRAIN_SIZE),:);
test_set = d1(permutation(:,TRAIN_SIZE+1:end),:);

% Part 7 
% Create classification trees using the training data, while
% experimenting with different pruning levels.
train_attr = train_set(:,1:4);
train_class = train_set(:,5);

tree5 = ClassificationTree.fit(train_attr, train_class, 'MinLeaf', 5);
tree25 = ClassificationTree.fit(train_attr, train_class, 'MinLeaf', 25);
tree50 = ClassificationTree.fit(train_attr, train_class, 'MinLeaf', 50);

view(tree5, 'Mode', 'Graph');
view(tree25, 'Mode', 'Graph');
view(tree50, 'Mode', 'Graph');

% Part 8 
% Testing the trees and confusion matricies report
test_attr = test_set(:,1:4);
test_class = test_set(:,5);

test_results_5 = tree5.predict(test_attr);
confusion_5 = confusionmat(test_class, test_results_5);

test_results_25 = tree25.predict(test_attr);
confusion_25 = confusionmat(test_class, test_results_25);

test_results_50 = tree50.predict(test_attr);
confusion_50 = confusionmat(test_class, test_results_50);