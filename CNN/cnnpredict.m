function [predictions_array] = cnnpredict(trained_model, test_images)
  
  prediction_time_0 = tic; %start timer

  batchsize = size(trained_model.layers{1}.a{1}, 3);
  num_test_data_samples = size(test_images, 3);
  
  predictions_array = []; %final size will be padded to a complete multiple of batchsize
  
  if  num_test_data_samples <= batchsize
    
   test_images = add_dummy_padding_data(test_images, batchsize);
   out_model = cnnff(trained_model, test_images);
   predictions_array = out_model.o(:,1:batchsize);
   
  elseif  num_test_data_samples > batchsize
   
    batch_cell_of_images = split_images_into_batches(test_images, batchsize, num_test_data_samples);
    for batch_id = 1 : size(batch_cell_of_images(),2)
      test_images = batch_cell_of_images{batch_id};
      out_model = cnnff(trained_model, test_images);
      current_predictions_array = out_model.o(:,1:batchsize);
      predictions_array = [predictions_array current_predictions_array];
    end
  end 
  
  predictions_array =  predictions_array(: ,1:num_test_data_samples); %final reduction of the predition array to match the number of test samples
  total_prediction_time = toc(prediction_time_0); %stop timer
  display(['Total elapsed prediction time ', num2str(total_prediction_time)]); % Display elapsed time
endfunction


%% fuction for padding an array of images to the size of a complete batch
function padded_test_images = add_dummy_padding_data(test_images, batchsize);
  [r,c, num_samples] = size(test_images);
  if num_samples == batchsize
    padded_test_images = test_images;
  else
    padded_test_images(:, :, 1:num_samples) = test_images; %copy of the valid input data 
    missing_range = (num_samples + 1) : batchsize;
    for i = missing_range
      padded_test_images(:, :, i) = zeros(r, c); %dummy black images
    end
  end
endfunction

%%function generating an array full of individual batches of data from an array of images with arbitrary length
function batch_cell_of_images = split_images_into_batches(test_images, batchsize, num_test_data_samples);
  
 num_batches = ceil(num_test_data_samples /batchsize);
 
 counter = 0;
 for batch_id = 1 : num_batches - 1  
   curent_batch = test_images(:, :, 1 + counter: batchsize*batch_id);
   counter = counter + batchsize;
   batch_cell_of_images{batch_id} = curent_batch;
 end 
  
  rest_range = (num_test_data_samples +1 - mod(num_test_data_samples, batchsize)) : num_test_data_samples;
  remaining_test_images = test_images(:, :, rest_range);
  batch_cell_of_images{num_batches} = add_dummy_padding_data(remaining_test_images, batchsize);

endfunction