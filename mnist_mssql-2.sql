-- creating a database first 
create database mnist_nn;

-- connecting to a database
use mnist_nn;

-- DO NOT NEED TO RUN THIS at first; run this only when you want to rerun the experiment(before rerunning the TrainNetwork procedure) 
-- since we need to initialize our weights again. 
-- so after dropping these tables, create them again and bulk insert data into those again. I have fixed values for the initial weights
-- in the txt files.  
-- do not drop training_data and training_labels tables because we never make changes to those tables. 
DROP TABLE layer1_weights;
DROP TABLE layer2_weights;
DROP TABLE layer1_biases;
DROP TABLE layer2_biases;
drop table min_loss_value; -- this table also gets changed within procedure, therefore initialize its value too.

-- creating two tables that hold training data information and 4 tables for holding weights 
CREATE TABLE training_data(RowIndex INT, ColumnIndex INT, CellValue INT);
CREATE TABLE training_labels(RowIndex INT, ColumnIndex INT, Class INT);
CREATE TABLE layer1_weights(RowIndex INT, ColumnIndex INT, CellValue DECIMAL(10,7));
CREATE TABLE layer2_weights(RowIndex INT, ColumnIndex INT, CellValue DECIMAL(10,7));
CREATE TABLE layer1_biases(RowIndex INT, ColumnIndex INT, CellValue DECIMAL(10,7));
CREATE TABLE layer2_biases(RowIndex INT, ColumnIndex INT, CellValue DECIMAL(10,7));

-- inserting values into the tables from txt files
-- weights are initialized using Xavier algorithm  
BULK INSERT training_data FROM 'C:\Users\v-aiaina\Fall2017_Research_Aiko\New experiments and new data\training_data_mnist1.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); --- loaded!
BULK INSERT training_data FROM 'C:\Users\v-aiaina\Fall2017_Research_Aiko\New experiments and new data\training_data_mnist2.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); --- loaded!
BULK INSERT training_data FROM 'C:\Users\v-aiaina\Fall2017_Research_Aiko\New experiments and new data\training_data_mnist3.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); --- loaded! 
BULK INSERT training_data FROM 'C:\Users\v-aiaina\Fall2017_Research_Aiko\New experiments and new data\training_data_mnist4.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); --- loaded!
BULK INSERT training_data FROM 'C:\Users\v-aiaina\Fall2017_Research_Aiko\New experiments and new data\training_data_mnist5.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); --- loaded!
BULK INSERT training_data FROM 'C:\Users\v-aiaina\Fall2017_Research_Aiko\New experiments and new data\training_data_mnist6.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); --- ! 

bulk INSERT training_labels from 'C:/Users/v-aiaina/Downloads/training_labels_mnist.txt' with (fieldterminator = ',', rowterminator = '0x0a');
bulk INSERT layer1_weights from 'C:/Users/v-aiaina/Downloads/layer1_weights_mnist.txt' with (fieldterminator = ',', rowterminator = '0x0a');
bulk INSERT layer2_weights from 'C:/Users/v-aiaina/Downloads/layer2_weights_mnist.txt' with (fieldterminator = ',', rowterminator = '0x0a');
bulk INSERT layer1_biases from 'C:/Users/v-aiaina/Downloads/layer1_biases_mnist.txt' with (fieldterminator = ',', rowterminator = '0x0a');
bulk INSERT layer2_biases from 'C:/Users/v-aiaina/Downloads/layer2_biases_mnist.txt' with (fieldterminator = ',', rowterminator = '0x0a');

-- this is a table that holds only one value -> minimum cross entropy loss throughout all of the iterations in the training
-- we update this value for the epoch that has loss value less than the min we've had so far. 
create table min_loss_value(EpochNumber INT, Loss DECIMAL(10,5));
insert into min_loss_value values (0, 10000.0); -- initialize loss to a very large value at first.

-- just testing if data was loaded correctly 
select count(*) from layer1_biases;  --         50 rows
select count(*) from layer2_biases;  --         10
select count(*) from layer2_weights; --        500
select count(*) from layer1_weights; --     39 200 
select count(*) from training_data;  -- 47 040 000 
select count(*) from training_labels;--    600 000 

-- Training procedure! 
-- Input data for this training procedure is called MNIST dataset. 
-- It contains 60,000 images of size 28*28 and tries to predict 10 digits from 0-9.
-- this training procedure accepts two arguments: 
-- ThresholdError = the cross entropy loss value we would like to reach 
-- Step = learning rate value (used for updating weights) 
-- The structure of the neural network trained here is the following:
-- 1) it has three layers - input, hidden, output layer
-- 2) this is an implementation of the gradient descent algorithm - so we update weights once for the pass through the entire training dataset
-- 3) there are 784 nodes in the input layer (due to input data), 50 nodes in the hidden (chosen through some experiments), 10 nodes in the output
-- (because of the number of labels we need to predict) 
IF OBJECT_ID (N'TrainNetwork', N'P') IS NOT NULL
    DROP PROCEDURE TrainNetwork;
GO
CREATE PROCEDURE TrainNetwork(@ThresholdError DECIMAL(10,5), @Step DECIMAL(10, 7))
AS
BEGIN
    DECLARE @i INT
    SET @i = 1

	-- So for now I am using a desired threshold value to stop the training procedure 
	WHILE (select loss from min_loss_value) > @ThresholdError 
        BEGIN
        SET @i = @i + 1

		-- dropping tables that are created in each iteration of the loop, store intermediate results
        IF OBJECT_ID ('hidden_layer', 'U') IS NOT NULL
        DROP TABLE hidden_layer
        IF OBJECT_ID ('output_layer', 'U') IS NOT NULL
        DROP TABLE output_layer
        if OBJECT_ID ('output_probs', 'U') Is not null 
        drop table output_probs
        IF OBJECT_ID ('output_error_signal', 'U') IS NOT NULL
        DROP TABLE output_error_signal
        IF OBJECT_ID ('error_signal_hidden', 'U') IS NOT NULL
        DROP TABLE error_signal_hidden

        -- hidden_layer table contains values of tanh activation function as the output from the hidden layer
		-- the nested code inside finds net value for the hidden layer and the outer part applies activation function to net value.
		select * into hidden_layer from
        (SELECT input_layer.RowIndex, input_layer.ColumnIndex, 
        (exp(input_layer.net+layer1_biases.CellValue)-exp(-(input_layer.net+layer1_biases.CellValue)))/
        (exp(input_layer.net+layer1_biases.CellValue)+exp(-(input_layer.net+layer1_biases.CellValue))) AS tanh_output
        FROM 
		(SELECT training_data.RowIndex, layer1_weights.ColumnIndex, sum(training_data.CellValue*layer1_weights.CellValue) AS net 
		FROM training_data CROSS JOIN layer1_weights WHERE training_data.ColumnIndex = layer1_weights.RowIndex 
        GROUP BY training_data.RowIndex, layer1_weights.ColumnIndex) as input_layer, layer1_biases 
        WHERE input_layer.ColumnIndex = layer1_biases.ColumnIndex) AS t2;

		--print 'done with hidden layer'; 

		-- table output_layer contains net values for the last output layer, 
		-- but without applying activation function yet (softmax activation should be applied to these in the next query)
		SELECT * INTO output_layer FROM 
        (SELECT t3.RowIndex, t3.ColumnIndex, t3.net2+layer2_biases.CellValue AS net2 FROM 
        layer2_biases,
        (SELECT hidden_layer.RowIndex, layer2_weights.ColumnIndex, sum(hidden_layer.tanh_output*layer2_weights.CellValue) AS
        net2 FROM
        hidden_layer CROSS JOIN layer2_weights WHERE hidden_layer.ColumnIndex = layer2_weights.RowIndex
        GROUP BY hidden_layer.RowIndex, layer2_weights.ColumnIndex) AS t3
        WHERE layer2_biases.ColumnIndex=t3.ColumnIndex) AS t4;

		--print 'done with output layer';

		-- table output_probs contains output results for the last output layer, so results that come from applying softmax activation function
        SELECT * into output_probs from 
        (select output_layer.RowIndex, output_layer.ColumnIndex, 
        (exp(output_layer.net2)/t2.sum_exp) as softmax_output from output_layer, 
		(select output_layer.RowIndex, sum(exp(output_layer.net2)) as sum_exp from output_layer group by 
		output_layer.RowIndex) as t2 where output_layer.RowIndex = t2.RowIndex) as t5; 

		--print 'done with output probs';

		-- now we compute cross entropy loss value, this loss value is compared to the min_loss_value variable 
		-- if our new loss value is smaller than min_loss_value then we save the new min value to the variable 
		-- otherwise we do not update that value; 
		update min_loss_value
		set EpochNumber = @i, loss = case when loss > new_computed_loss THEN new_computed_loss ELSE loss end from 
		(select sum(-training_labels.Class*LOG(softmax_output))/max(training_labels.RowIndex) as new_computed_loss from output_probs, training_labels
        where training_labels.RowIndex = output_probs.RowIndex and training_labels.ColumnIndex = output_probs.ColumnIndex) as tt; 

		--print 'done updating minlossvalue';

		-- this table saves errors for the layer 2 (output layer)
		-- I need to create some variable for the number of instances in the input, I have 60,000 hardcoded for now  
		-- maybe this value can be also passed as an argument to the procedure
		SELECT * INTO output_error_signal FROM 
        (SELECT output_probs.RowIndex, output_probs.ColumnIndex,(output_probs.softmax_output-training_labels.Class)/60000    
        AS error_value
        FROM training_labels CROSS JOIN output_probs 
        WHERE training_labels.RowIndex = output_probs.RowIndex AND training_labels.ColumnIndex = output_probs.ColumnIndex) AS 
        t8;

		--print 'done with output_error_signal';

        -- these are errors for layer 1 (hidden layer errors)
		-- make use of the derivative of tanh function
        SELECT * INTO error_signal_hidden FROM 
        (SELECT t9.RowIndex, t9.ColumnIndex, (t9.error_value * (1-SQUARE(hidden_layer.tanh_output))) AS       
        error_value
        FROM hidden_layer,
        (SELECT output_error_signal.RowIndex, layer2_weights.RowIndex AS ColumnIndex, sum(output_error_signal.error_value*  
        layer2_weights.CellValue) AS error_value FROM 
        output_error_signal CROSS JOIN layer2_weights 
        WHERE output_error_signal.ColumnIndex = layer2_weights.ColumnIndex
        GROUP BY output_error_signal.RowIndex, layer2_weights.RowIndex) AS t9
        WHERE hidden_layer.ColumnIndex = t9.ColumnIndex AND hidden_layer.RowIndex = t9.RowIndex) AS t10; 

		--print 'done with error signal hidden';

        -- updating weights for Layer 2
		UPDATE layer2_weights
        SET CellValue = CellValue - @Step*gradient_layer2_weights.gradient_value
        FROM layer2_weights, (SELECT hidden_layer.ColumnIndex AS RowIndex, output_error_signal.ColumnIndex, 
        sum(hidden_layer.tanh_output*output_error_signal.error_value) AS gradient_value
        FROM hidden_layer CROSS JOIN output_error_signal 
        WHERE hidden_layer.RowIndex = output_error_signal.RowIndex
        GROUP BY hidden_layer.ColumnIndex, output_error_signal.ColumnIndex) as gradient_layer2_weights 
        WHERE layer2_weights.RowIndex = gradient_layer2_weights.RowIndex AND layer2_weights.ColumnIndex = 
        gradient_layer2_weights.ColumnIndex;

		--print 'done updating layer2_weights';

        -- updating weights of biases for Layer 2
		UPDATE layer2_biases 
        SET CellValue = CellValue - @Step*gradient_layer2_bias.gradient_value
        FROM layer2_biases, (SELECT 1 AS RowIndex, output_error_signal.ColumnIndex, sum(output_error_signal.error_value) AS 
        gradient_value FROM
        output_error_signal 
        GROUP BY output_error_signal.ColumnIndex) as gradient_layer2_bias
        WHERE layer2_biases.RowIndex = gradient_layer2_bias.RowIndex AND layer2_biases.ColumnIndex =    
        gradient_layer2_bias.ColumnIndex;

		--print 'done updating layer2_biases';

        -- updating weights for Layer 1
		UPDATE layer1_weights
        SET CellValue = CellValue - @Step*gradient_layer1_weights.gradient_value
        FROM layer1_weights, (SELECT training_data.ColumnIndex AS RowIndex, error_signal_hidden.ColumnIndex, 
        sum(training_data.CellValue*error_signal_hidden.error_value) AS gradient_value
        FROM training_data CROSS JOIN error_signal_hidden
        WHERE training_data.RowIndex = error_signal_hidden.RowIndex
        GROUP BY training_data.ColumnIndex, error_signal_hidden.ColumnIndex) as gradient_layer1_weights 
        WHERE layer1_weights.RowIndex = gradient_layer1_weights.RowIndex AND layer1_weights.ColumnIndex = 
        gradient_layer1_weights.ColumnIndex;

		--print 'done updating layer1_weights';

        -- updating weights of biases for Layer 1
		UPDATE layer1_biases 
        SET CellValue = CellValue - @Step*gradient_layer1_bias.gradient_value
        FROM layer1_biases, (SELECT 1 AS RowIndex, error_signal_hidden.ColumnIndex, sum(error_signal_hidden.error_value) AS 
        gradient_value FROM 
        error_signal_hidden 
        GROUP BY error_signal_hidden.ColumnIndex) as gradient_layer1_bias
        WHERE layer1_biases.RowIndex = gradient_layer1_bias.RowIndex AND layer1_biases.ColumnIndex = 
        gradient_layer1_bias.ColumnIndex;

		--print 'done updating layer1_biases';
		--print 'done with an epoch!'; 

-- while loop ends! 
    END
-- procedure ends 
END
GO

-- calling trainnetwork procedure and passing two arguments to it
DECLARE @stepss INT
SET @stepss = 3   -- this value was found during experiments, gives good accuracy
DECLARE @threshold DECIMAL(10,5)
SET @threshold = 0.130 -- this threshold value means that we want to reach accuracy around -log(0.130) = 0.88; i.e. 88%
-- we can definitely increase this threshold value if we want to have less # of epochs; after first epoch loss value goes down to 2.3
EXEC TrainNetwork @threshold, @stepss
GO

-- PREDICTION PHASE 
-- create tables to hold test data, test data size is 10,000 images
CREATE TABLE test_data(RowIndex INT, ColumnIndex INT, CellValue INT);
CREATE TABLE test_labels(RowIndex INT, ColumnIndex INT, Class INT);

-- insert values from txt files
bulk INSERT test_data from 'C:/Users/v-aiaina/Downloads/test_data_mnist.txt' with (fieldterminator = ',', rowterminator = '0x0a');
bulk INSERT test_labels from 'C:/Users/v-aiaina/Downloads/test_labels_mnist.txt' with (fieldterminator = ',', rowterminator = '0x0a');

-- FORWARD PROPAGATION HAPPENS ONCE for test data based on weights computed in the trainnetwork procedure 
-- this computes tanh output for hidden layer
IF OBJECT_ID ('hidden_layer_testdata', 'U') IS NOT NULL
DROP TABLE hidden_layer_testdata

select * into hidden_layer_testdata from
(SELECT input_layer.RowIndex, input_layer.ColumnIndex, 
(exp(input_layer.net+layer1_biases.CellValue)-exp(-(input_layer.net+layer1_biases.CellValue)))/
(exp(input_layer.net+layer1_biases.CellValue)+exp(-(input_layer.net+layer1_biases.CellValue))) AS tanh_output
FROM (SELECT test_data.RowIndex, layer1_weights.ColumnIndex, sum(test_data.CellValue*layer1_weights.CellValue) AS
net FROM test_data CROSS JOIN layer1_weights WHERE test_data.ColumnIndex = layer1_weights.RowIndex 
GROUP BY test_data.RowIndex, layer1_weights.ColumnIndex) as input_layer, layer1_biases 
WHERE input_layer.ColumnIndex = layer1_biases.ColumnIndex) AS t2;

-- table output_layer contains net values for the last output layer, 
-- but without applying activation function yet (softmax activation should be applied to these in the next query)
IF OBJECT_ID ('output_layer_testdata', 'U') IS NOT NULL
DROP TABLE output_layer_testdata

SELECT * INTO output_layer_testdata FROM 
(SELECT t3.RowIndex, t3.ColumnIndex, t3.net2+layer2_biases.CellValue AS net2 FROM layer2_biases,
(SELECT hidden_layer_testdata.RowIndex, layer2_weights.ColumnIndex, sum(hidden_layer_testdata.tanh_output*layer2_weights.CellValue) AS net2 
FROM hidden_layer_testdata CROSS JOIN layer2_weights WHERE hidden_layer_testdata.ColumnIndex = layer2_weights.RowIndex
GROUP BY hidden_layer_testdata.RowIndex, layer2_weights.ColumnIndex) AS t3
WHERE layer2_biases.ColumnIndex=t3.ColumnIndex) AS t4;

-- table output_probs contains output results for the last output layer, so results that come from applying softmax activation function
IF OBJECT_ID ('output_probs_testdata', 'U') IS NOT NULL
DROP TABLE output_probs_testdata

SELECT * into output_probs_testdata from 
(select output_layer_testdata.RowIndex, output_layer_testdata.ColumnIndex, 
(exp(output_layer_testdata.net2)/t2.sum_exp) as softmax_output from output_layer_testdata, 
(select output_layer_testdata.RowIndex, sum(exp(output_layer_testdata.net2)) as sum_exp from output_layer_testdata group by 
output_layer_testdata.RowIndex) as t2 where output_layer_testdata.RowIndex = t2.RowIndex) as t5;

-- let's compute loss value for the test data; result is just printed out  
select sum(-test_labels.Class*LOG(softmax_output))/max(test_labels.RowIndex) as loss from output_probs_testdata, test_labels
where test_labels.RowIndex = output_probs_testdata.RowIndex and test_labels.ColumnIndex = output_probs_testdata.ColumnIndex;

-- now we compute the accuracy of our predictions; 
-- in order to do that we check values in the output_probs_testdata and for each RowIndex we decide 
-- what was predicted; this just prints number of correctly predicted labels for total of 10,000 images from test data
select count(*) from 
(select RowIndex, ColumnIndex, max(softmax_output) as max_value_softmax from output_probs_testdata
group by RowIndex) test1 cross join   
(select RowIndex, ColumnIndex, max(Class) as max_value_class from test_labels
group by RowIndex) test2 
where test1.RowIndex = test2.RowIndex AND test1.ColumnIndex = test2.ColumnIndex;   
