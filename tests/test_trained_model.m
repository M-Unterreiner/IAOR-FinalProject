function test_trained_model
  
  %load trained model
  %TODO 
  % Exaple: load('trained_models\my_network.mat');
  
  %the model name 'cnn' was defined by the user in the starter_example
  fields = fieldnames(cnn) %display information about the loaded model;  

  %load test image data set containing images and labels (if available)
  %you can choose which test data you want to use
  %TODO
  
  num_classes = 26; %All letters in the English alphabet
  num_test_samples = 10; %
  
  %generate array with prediction vectors of size num_classes -by- num_test_samples (e.g. 26-by-10 for 10 test images)
  %TODO

  % visualize same of the predictions
  %TODO
  
end