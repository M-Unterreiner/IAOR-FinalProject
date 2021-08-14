%%------- SAMPLE IMAGES DISPLAY ------- 
function visualize_sample_data(images, labels, num_samples_to_show, total_num_samples, figure_name, num_classes)
   figure( 'name', figure_name, 'numbertitle', 'off'); 
   for k = 1: num_samples_to_show
    current_index = round((rand()*(total_num_samples -1.0)) +1);
    #test_x_sample = reshape(images(current_index, :), 28, 28);
    test_x_sample = images(:,:,current_index);
    true_label = [~, true_label] = max(labels(:,current_index));
    subplot(1, num_samples_to_show,k,"align" );
    if num_classes >10
     imshow(test_x_sample, []); %no transposition needed for the EMNIST images
    else
     imshow(test_x_sample', []);% transposition needed for the MNIST images
    end
    title( strcat('class: ', map_label_index_to_string(true_label,num_classes)));
     %labels(:, current_index);
     %true_label
  end 
  pause (0.05);
  
end