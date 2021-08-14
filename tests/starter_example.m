function starter_example
t0 = tic;
load("data/mnist_uint8"); %load MNIST data set


num_classes = size(train_y, 2);
max_train_samples_per_class = size(train_x, 1);
max_test_samples_per_class = size(test_x, 1);


%training and test images 
train_x = double(reshape(train_x',28,28,max_train_samples_per_class))/255;
test_x = double(reshape(test_x',28,28,max_test_samples_per_class))/255;


%training and test label data
train_y = double(train_y');
test_y = double(test_y');


%reduce data samples
skip_number = 10; 
train_x = train_x(:,:,1:skip_number:max_train_samples_per_class);
test_x = test_x(:,:,1:skip_number:max_test_samples_per_class);
train_y = train_y(:,1:skip_number:max_train_samples_per_class);
test_y = test_y(:,1:skip_number:max_test_samples_per_class);


total_num_samples = size(train_x, 1);
num_samples_to_show = 4;
visualize_sample_data(train_x, train_y, num_samples_to_show, total_num_samples, 'SAMPLE DATA', num_classes);

%% Train a i-6c-2s-12c-2s-o Convolutional neural network 

%Network Atchitecture
conv_kernel_size = 5;
base_feature_map = 6;

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', base_feature_map, 'kernelsize', conv_kernel_size) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', base_feature_map*2, 'kernelsize', conv_kernel_size) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

%network parameters
opts.alpha = 0.6; %learning rate
opts.batchsize = 50; %num samples per batch
opts.numepochs = 5; %num training epochs

cnn = cnnsetup(cnn, train_x, train_y); 
cnn = cnntrain(cnn, train_x, train_y, opts, test_x, test_y); %train the model
%cnn = cnntrain(cnn, train_x, train_y, opts); %train without computing test error after each epoch

%% Note on function 'cnntrain': 
%% The last two parameters (test_x and test_y) are optional. 
%% Calling the fumction 'cnntrain' without these parameters will
%% skip the plotting of train vs test errors, which will result 
%% in quicker execution.


[er, bad] = cnntest(cnn, test_x, test_y); % test trained model

if er < 0.012 
  save -mat7-binary trained_starter_model.mat cnn % exmple how to save the trained model for later use
else
  disp(strcat('Final Test Error= ',  num2str(er)));
end


%plot mean squared error
figure; plot(cnn.rL); title('Loss fuction');  xlabel ('number of processed batches'),  ylabel ('error');

%%visualize predictions for test samples
num_samples = 2;
% the cnnpredict function returns an array with vectors containing the class probability distribution for each test sample
predictions_array = cnnpredict(cnn, test_x(:, :, 1:num_samples)); % Here 'predictions_array' will be of size 10-by-num_samples
visualize_prediction_results(test_x, predictions_array, num_samples, test_y); % test_y is an optional parameter and can be omitted 

%test once more with test image not included in the data set
my_test_img = imread('data/Images/task_B/1_simple.JPG')';
predictions_array = cnnpredict(cnn, my_test_img); % Here 'predictions_array' will be of size 10-by-1 because we test only one sample
visualize_prediction_results(my_test_img, predictions_array, 1);


total_elapsed_time = toc(t0);
disp(strcat('total elaplsed time= ', num2str(total_elapsed_time), 'seconds'));

end