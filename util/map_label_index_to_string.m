%%------- CHARACTER CONVERSION ------- 
function string_result = map_label_index_to_string(numeric_value, num_classes)
  if num_classes > 10
    string_result = char( numeric_value + 64); % ASCII conversion: A-Z -> 65-90 
  else
    string_result = num2str(numeric_value - 1); % 0-9 -> '0'-'9'
  end
end