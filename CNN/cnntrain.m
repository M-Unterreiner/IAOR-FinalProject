function net = cnntrain(net, x, y, opts, x_test, y_test)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    test_er = -1;
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            
        end  
        train_er_after_each_epoc(i) = net.rL(end);      
        if exist('x_test', 'var') && exist('y_test', 'var')
          disp(' ---- Computing test error ---');
          [er, bad] = cnntest(net, x_test, y_test); % test trained model
          test_er(i) = er;
          %disp(strcat("Current last rL value",num2str(net.rL(end))));
        end
        toc;
    end
    
    if test_er != -1
      figure; title('train error vs test error at the end of each epoch');
      plot(test_er, ';test error;', train_er_after_each_epoc, ';train error;');
       xlabel ('epoch number'),  ylabel ('error');
    end
    
end
