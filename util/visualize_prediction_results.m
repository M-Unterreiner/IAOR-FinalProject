function visualize_prediction_results(my_test_img, predictions_array, num_samples_to_show, test_y)
  
  num_classes = size(predictions_array, 1);
  
  % visualize same of the predictions
  n =  min(num_samples_to_show, size(my_test_img,3)); %show first n  predictions
  for  sample_index = 1: n
    
    [prediction_value, index] = max(predictions_array(:, sample_index));
     predicted_label = map_label_index_to_string(index, num_classes); 
     if exist('test_y', 'var')
       [~, true_label_index] = max(test_y(:, sample_index));
       true_label = map_label_index_to_string(true_label_index, num_classes);
     else
       true_label ='NOT KNOWN';
        #true_label ='A';
     end
     figure;
     if num_classes >10
      imshow(my_test_img(:, :, sample_index), []); %no transposition needed for the EMNIST images
     else
      imshow(my_test_img(:, :, sample_index)', []); %transposition needed for the MNIST images
     end
     title( strcat(' *  predicted class : ', predicted_label, ' * ', ...
                   ' *  confidence : ', num2str(prediction_value*100), "%", ' * ',...
                   ' *  true class : ', true_label, ' * '));
  end
  
endfunction
