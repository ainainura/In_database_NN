-- creating database 
CREATE DATABASE mnist_nn_hekaton ON PRIMARY(NAME = SqlHints_DATA, 
FILENAME = '~\SqlHints_Data.mdf'); 

-- just checking some information here
SELECT name, compatibility_level FROM sys.databases; 

-- add a filegroup to a database otherwise you can't use memory-optimized tables 
ALTER DATABASE mnist_nn_hekaton ADD filegroup SqlHints_XTP_FG CONTAINS MEMORY_OPTIMIZED_DATA;

-- add file to a filegroup 
ALTER DATABASE mnist_nn_hekaton
ADD FILE(NAME = 'SqlHints_XTP_CHKPOINT', 
FILENAME = '~\SqlHints_XTP_CHKPOINT') 
TO FILEGROUP SqlHints_XTP_FG;

-- connect to a db
USE mnist_nn_hekaton;

-- DO NOT RUN! this is necessary for only rerunning the experiments, when we want to rerun TrainNetwork procedure.
DROP TABLE layer2_biases;
DROP TABLE layer2_weights;
DROP TABLE layer1_weights;
DROP TABLE layer1_biases;
drop table dbo.min_loss_value;

-- creating two tables that hold training data information and 4 tables for holding weights; 
-- all of them are memory-optimized tables 
CREATE TABLE training_data(RowIndex INT, ColumnIndex INT, CellValue INT, 
			   PRIMARY KEY NONCLUSTERED (RowIndex, ColumnIndex)) WITH (MEMORY_OPTIMIZED = ON, 
										   DURABILITY = SCHEMA_AND_DATA); 
CREATE TABLE training_labels(RowIndex INT, ColumnIndex INT, Class INT, 
			     PRIMARY KEY NONCLUSTERED (RowIndex, ColumnIndex)) WITH (MEMORY_OPTIMIZED = ON, 
										     DURABILITY = SCHEMA_AND_DATA);
CREATE TABLE layer1_weights(RowIndex INT, ColumnIndex INT, CellValue DECIMAL(10,7), 
			    PRIMARY KEY NONCLUSTERED (RowIndex, ColumnIndex)) WITH (MEMORY_OPTIMIZED = ON, 
										    DURABILITY = SCHEMA_AND_DATA);
CREATE TABLE layer2_weights(RowIndex INT, ColumnIndex INT, CellValue DECIMAL(10,7), 
			    PRIMARY KEY NONCLUSTERED (RowIndex, ColumnIndex)) WITH (MEMORY_OPTIMIZED = ON, 
										    DURABILITY = SCHEMA_AND_DATA);
CREATE TABLE layer1_biases(RowIndex INT, ColumnIndex INT, CellValue DECIMAL(10,7), 
			   PRIMARY KEY NONCLUSTERED (RowIndex, ColumnIndex)) WITH (MEMORY_OPTIMIZED = ON, 
										   DURABILITY = SCHEMA_AND_DATA);
CREATE TABLE layer2_biases(RowIndex INT, ColumnIndex INT, CellValue DECIMAL(10,7), 
			   PRIMARY KEY NONCLUSTERED (RowIndex, ColumnIndex)) WITH (MEMORY_OPTIMIZED = ON, 
										   DURABILITY = SCHEMA_AND_DATA);

-- table which will save minimum loss value during the procedure runtime, also memory optimized
CREATE TABLE dbo.min_loss_value(EpochNumber INT, Loss DECIMAL(10,5), 
				PRIMARY KEY NONCLUSTERED (EpochNumber)) WITH (MEMORY_OPTIMIZED = ON, 
									      DURABILITY = SCHEMA_AND_DATA);
INSERT INTO dbo.min_loss_value VALUES (0, 10000.0);
SELECT * FROM min_loss_value;
--delete from min_loss_value where EpochNumber = 0;


-- loading input data and initial weights of the neural net; training data was loaded through 6 different files.
BULK INSERT training_data FROM '~\training_data_mnist1.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); 
BULK INSERT training_data FROM '~\training_data_mnist2.txt' WITH (fieldterminator = ',', rowterminator = '0x0a');
BULK INSERT training_data FROM '~\training_data_mnist3.txt' WITH (fieldterminator = ',', rowterminator = '0x0a');  
BULK INSERT training_data FROM '~\training_data_mnist4.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); 
BULK INSERT training_data FROM '~\training_data_mnist5.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); 
BULK INSERT training_data FROM '~\training_data_mnist6.txt' WITH (fieldterminator = ',', rowterminator = '0x0a'); 
 
BULK INSERT training_labels FROM '~\training_labels_mnist.txt' WITH (fieldterminator = ',', rowterminator = '0x0a');
BULK INSERT layer1_weights FROM '~\layer1_weights_mnist.txt' WITH (fieldterminator = ',', rowterminator = '0x0a');
BULK INSERT layer2_weights FROM '~\layer2_weights_mnist.txt' WITH (fieldterminator = ',', rowterminator = '0x0a');
BULK INSERT layer1_biases FROM '~\layer1_biases_mnist.txt' WITH (fieldterminator = ',', rowterminator = '0x0a');
BULK INSERT layer2_biases FROM '~\layer2_biases_mnist.txt' WITH (fieldterminator = ',', rowterminator = '0x0a');

-- just testing correctness of loading the data
SELECT count(*) FROM training_data;  
SELECT count(*) FROM training_labels;
SELECT count(*) FROM layer1_biases;
select count(*) from layer2_biases;
select count(*) from layer1_weights;
select count(*) from layer2_weights;

-- this is necessary for the update procedure (there is a limitation in natively compiled procedures 
							  -- that UPDATE .. FROM .. doesn't work)
-- and also for initializing some variable tables which store intermediate results.
-- both of these are memory optimized too. Basically natively compiled procedure cannot work with other types of objects,
-- they all should be memory optimized. 
DROP TYPE IF EXISTS generic_table_type;
CREATE TYPE dbo.generic_table_type AS TABLE  
	(  
	RowIndex       INT,  
	ColumnIndex    INT,
	CellValue	   DECIMAL(10,5),
	PRIMARY KEY NONCLUSTERED (RowIndex, ColumnIndex)  
	)   
	WITH (MEMORY_OPTIMIZED = ON, DURABILITY = SCHEMA_ONLY);  

DROP TYPE IF EXISTS type_for_min_loss;
CREATE TYPE dbo.type_for_min_loss AS TABLE
(
Loss DECIMAL(10,5) PRIMARY KEY NONCLUSTERED
)
WITH (MEMORY_OPTIMIZED = ON, DURABILITY = SCHEMA_ONLY);  

-- TRAIN NETWORK - NATIVELY COMPILED STORED PROCEDURE!
-- 
-- Input data for this training procedure is called MNIST dataset. 
-- It contains 60,000 images of size 28*28 and tries to predict 10 digits from 0-9.
-- this training procedure accepts two arguments: 
-- ThresholdError = the cross entropy loss value we would like to reach 
-- Step = learning rate value (used for updating weights) 
-- The structure of the neural network trained here is the following:
-- 1) it has three layers - input, hidden, output layer
-- 2) this is an implementation of the gradient descent algorithm - so we 
				-- update weights once for the pass 
			        -- through the entire training dataset
-- 3) there are 784 nodes in the input layer (due to input data), 
				-- 50 nodes in the hidden (chosen through some experiments), 
				-- 10 nodes in the output (because of the number of labels we need to predict) 
IF OBJECT_ID (N'TrainNetwork', N'P') IS NOT NULL
    DROP PROCEDURE TrainNetwork;
GO
CREATE PROCEDURE dbo.TrainNetwork(@ThresholdError DECIMAL(10,5), @Step DECIMAL(10, 7)) 
									       WITH NATIVE_COMPILATION, SCHEMABINDING 
AS BEGIN ATOMIC WITH (TRANSACTION ISOLATION LEVEL = SNAPSHOT, LANGUAGE = N'us_english')

    DECLARE @i INT
    SET @i = 1
	DECLARE @threshold DECIMAL(10,5) -- just creating this because i cannot do nested queries here (only select statements can be nested)
	SELECT  @threshold = loss from dbo.min_loss_value; 
	
	--declaration of variable tables to store intermediate results
	DECLARE @hidden_layer dbo.generic_table_type;
	DECLARE @output_layer dbo.generic_table_type;
	DECLARE @output_probs dbo.generic_table_type;
	DECLARE @output_error_signal dbo.generic_table_type;
	DECLARE @error_signal_hidden dbo.generic_table_type;
	DECLARE @tt dbo.type_for_min_loss;
	DECLARE @gradient_layer2_bias dbo.generic_table_type;
	DECLARE @gradient_layer2_weights dbo.generic_table_type;
	DECLARE @gradient_layer1_bias dbo.generic_table_type;
	DECLARE @gradient_layer1_weights dbo.generic_table_type;

	-- For now I am using a threshold value to stop the training procedure 
	WHILE @threshold > @ThresholdError 
        BEGIN
        SET @i = @i + 1

		-- this piece of code is problematic, TRUNCATE operator doesn't work in natively compiled proc
		-- because of that I am removing rows like this. 
		-- Also, this code is problematic because even though rows are deleted from these tables 
		-- memory is not deallocated, table size keeps growing with each while loop.
		-- Memory will be deallocated only when you drop the table, 
			-- but dropping variable tables doesn't exist/doesn't work. 
		-- I couldn't figure out how to do garbage collection. 
		DELETE FROM @hidden_layer where RowIndex>0;
		DELETE FROM @output_layer where RowIndex>0;
		DELETE FROM @output_probs where RowIndex>0;
		DELETE FROM @output_error_signal where RowIndex>0;
		DELETE FROM @error_signal_hidden where RowIndex>0;
		DELETE FROM @tt where loss>0;
		DELETE FROM @gradient_layer2_bias where RowIndex>0;
		DELETE FROM @gradient_layer2_weights where RowIndex>0;
		DELETE FROM @gradient_layer1_weights where RowIndex>0;
		DELETE FROM @gradient_layer1_bias where RowIndex>0;
		

        -- hidden_layer table contains values of tanh activation function as the output from the hidden layer
		-- the nested code inside finds net value for the hidden layer and the outer 
		--part applies activation function to net value.
		INSERT INTO @hidden_layer SELECT RowIndex, ColumnIndex, tanh_output from
		(SELECT input_layer.RowIndex, input_layer.ColumnIndex, 
        (exp(input_layer.net+dbo.layer1_biases.CellValue)-exp(-(input_layer.net+dbo.layer1_biases.CellValue)))/
        (exp(input_layer.net+dbo.layer1_biases.CellValue)+exp(-(input_layer.net+dbo.layer1_biases.CellValue))) 
								AS tanh_output
        FROM 
		(SELECT dbo.training_data.RowIndex, dbo.layer1_weights.ColumnIndex, 
		 sum(dbo.training_data.CellValue*dbo.layer1_weights.CellValue) AS 
		net 
		FROM dbo.training_data CROSS JOIN dbo.layer1_weights WHERE 
		 dbo.training_data.ColumnIndex = dbo.layer1_weights.RowIndex 
        GROUP BY dbo.training_data.RowIndex, dbo.layer1_weights.ColumnIndex) AS input_layer, dbo.layer1_biases 
        WHERE input_layer.ColumnIndex = dbo.layer1_biases.ColumnIndex) AS t2;

		-- table output_layer contains net values for the last output layer, 
		-- but without applying activation function yet (softmax activation should 
							      -- be applied to these in the next query)
		INSERT INTO @output_layer SELECT RowIndex,ColumnIndex,net2 FROM 
		(SELECT t3.RowIndex, t3.ColumnIndex, t3.net2+dbo.layer2_biases.CellValue AS net2 FROM 
        dbo.layer2_biases,
        (SELECT HL.RowIndex, dbo.layer2_weights.ColumnIndex, sum(HL.CellValue*dbo.layer2_weights.CellValue) AS
        net2 FROM
        @hidden_layer HL CROSS JOIN dbo.layer2_weights WHERE HL.ColumnIndex = dbo.layer2_weights.RowIndex
        GROUP BY HL.RowIndex, dbo.layer2_weights.ColumnIndex) AS t3
        WHERE dbo.layer2_biases.ColumnIndex=t3.ColumnIndex) AS t4;

		-- table output_probs contains output results for the last output layer, 
							      -- so results that come from applying softmax 
		--activation function
		INSERT INTO @output_probs SELECT RowIndex, ColumnIndex, softmax_output FROM 
		(SELECT OL.RowIndex, OL.ColumnIndex, 
        (exp(OL.CellValue)/t2.sum_exp) AS softmax_output FROM @output_layer OL, 
		(SELECT OL1.RowIndex, sum(exp(OL1.CellValue)) AS sum_exp FROM @output_layer OL1 GROUP BY OL1.RowIndex) 
					  AS t2 WHERE OL.RowIndex = t2.RowIndex) AS t5; 
		
		-- now we compute cross entropy loss value, this loss value is compared to the min_loss_value variable 
		-- if our new loss value is smaller than min_loss_value then we save the new min value to the variable 
		-- rewriting update statement for min_loss_value
		INSERT INTO @tt SELECT new_computed_loss FROM 
		 (SELECT sum(-dbo.training_labels.Class*LOG(OP.CellValue))/max(dbo.training_labels.RowIndex) 
							    AS new_computed_loss FROM 
		@output_probs OP, dbo.training_labels
        WHERE dbo.training_labels.RowIndex = OP.RowIndex AND dbo.training_labels.ColumnIndex = OP.ColumnIndex) as tt2;

		declare @computed_loss DECIMAL(10,5); 
		set @computed_loss = (SELECT Loss from @tt T);

		-- couldn't update EpochNumber here too because it's a primary key, and they cannot be updated.
		UPDATE dbo.min_loss_value
		SET loss = CASE WHEN loss > @computed_loss THEN @computed_loss ELSE loss END; 

		-- update our @threshold variable too
		SELECT @threshold = loss from dbo.min_loss_value;

		-- this table saves errors for the layer 2 (output layer)
		-- I need to create some variable for the number of instances in the input, I have 60,000 hardcoded for now  
		INSERT INTO @output_error_signal SELECT RowIndex, ColumnIndex, error_value FROM 
		(SELECT OP.RowIndex, OP.ColumnIndex,(OP.CellValue-dbo.training_labels.Class)/60000    
        AS error_value
        FROM dbo.training_labels CROSS JOIN @output_probs OP
        WHERE dbo.training_labels.RowIndex = OP.RowIndex AND dbo.training_labels.ColumnIndex = OP.ColumnIndex) AS 
        t8;
				
        -- these are errors for layer 1 (hidden layer errors)	
		INSERT INTO @error_signal_hidden SELECT RowIndex, ColumnIndex, error_value FROM 
		(SELECT t9.RowIndex, t9.ColumnIndex, (t9.error_value * (1-SQUARE(HL.CellValue))) AS       
        error_value
        FROM @hidden_layer HL,
        (SELECT OES.RowIndex, dbo.layer2_weights.RowIndex AS ColumnIndex, sum(OES.CellValue*  
        dbo.layer2_weights.CellValue) AS error_value FROM 
        @output_error_signal OES CROSS JOIN dbo.layer2_weights 
        WHERE OES.ColumnIndex = dbo.layer2_weights.ColumnIndex
        GROUP BY OES.RowIndex, dbo.layer2_weights.RowIndex) AS t9
        WHERE HL.ColumnIndex = t9.ColumnIndex AND HL.RowIndex = t9.RowIndex) AS t10; 

		-- Now the rest of the code from here has been rewritten because "UPDATE .. FROM .." doesn't work in 
		-- natively compiled procedures. So, I used a variable to hold values from one table and to use them in updating
		-- another table. I used loops to go through those tables. I saw similar implementation on the official website
		-- and they said this is the way to rewrite UPDATE .. FROM .. statement.
		-- updating weights for Layer 2
		-- creating another temp table for holding gradient values for layer 2
		INSERT INTO @gradient_layer2_weights SELECT RowIndex, ColumnIndex, gradient_value FROM 
		(SELECT HL.ColumnIndex AS RowIndex, OES.ColumnIndex, 
        sum(HL.CellValue*OES.CellValue) AS gradient_value
        FROM @hidden_layer HL CROSS JOIN @output_error_signal OES 
        WHERE HL.RowIndex = OES.RowIndex
        GROUP BY HL.ColumnIndex, OES.ColumnIndex) as tt3;

		-- NEW CODE FOR UPDATING LAYER2_WEIGHTS
		-- ================================================================================================================
		DECLARE  
        @m INT = 1,
		@n INT = 1,  
        @maxM INT = (SELECT max(RowIndex) from dbo.layer2_weights),
		@maxN INT = (SELECT max(ColumnIndex) from dbo.layer2_weights),
		@gradient_val DECIMAL(10,5);

		---- Loop as a workaround to simulate a cursor.
		---- Iterate over the rows in the memory-optimized table  
		---- variable and perform an update for each row.  

		WHILE @m <= @maxM
		BEGIN
			WHILE @n <= @maxN
			BEGIN
				SELECT @gradient_val = CellValue
				FROM @gradient_layer2_weights
				WHERE RowIndex = @m AND ColumnIndex = @n;

				UPDATE dbo.layer2_weights
				SET CellValue = CellValue - @Step * @gradient_val
				WHERE RowIndex = @m AND ColumnIndex = @n;

				SET @n += 1;
			END
			SET @m += 1;
		END
		-- ====================================================================================================================

		-- creating new table for layer2_biases_gradient
		INSERT INTO @gradient_layer2_bias SELECT RowIndex, ColumnIndex, gradient_value FROM
		(SELECT 1 AS RowIndex, OES.ColumnIndex, sum(OES.CellValue) AS 
        gradient_value FROM
        @output_error_signal OES 
        GROUP BY OES.ColumnIndex) as tt4;  

		-- NEW CODE FOR UPDATING LAYER2_BIASES 
		-- ================================================================================================================
		SET @m = 1;
		SET @n = 1;  
		SET @maxM = (SELECT max(RowIndex) from dbo.layer2_biases);
		SET @maxN = (SELECT max(ColumnIndex) from dbo.layer2_biases);

		---- Loop as a workaround to simulate a cursor.
		---- Iterate over the rows in the memory-optimized table  
		---- variable and perform an update for each row.  

		WHILE @m <= @maxM
		BEGIN
			WHILE @n <= @maxN
			BEGIN
				SELECT @gradient_val = CellValue
				FROM @gradient_layer2_bias
				WHERE RowIndex = @m AND ColumnIndex = @n;

				UPDATE dbo.layer2_biases
				SET CellValue = CellValue - @Step * @gradient_val
				WHERE RowIndex = @m AND ColumnIndex = @n;

				SET @n += 1;
			END
			SET @m += 1;
		END
		-- ====================================================================================================================
		
        -- updating weights for Layer 1
		-- variable table holds gradients for layer 1 weights
		INSERT INTO @gradient_layer1_weights SELECT RowIndex, ColumnIndex, gradient_value FROM
		(SELECT dbo.training_data.ColumnIndex AS RowIndex, ESH.ColumnIndex, 
        sum(dbo.training_data.CellValue*ESH.CellValue) AS gradient_value
        FROM dbo.training_data CROSS JOIN @error_signal_hidden ESH
        WHERE dbo.training_data.RowIndex = ESH.RowIndex
        GROUP BY dbo.training_data.ColumnIndex, ESH.ColumnIndex) as tt5;

		-- NEW CODE FOR UPDATING LAYER1_WEIGHTS 
		-- ================================================================================================================
        SET @m = 1;
		SET @n = 1;  
        SET @maxM = (SELECT max(RowIndex) from dbo.layer1_weights);
		SET @maxN = (SELECT max(ColumnIndex) from dbo.layer1_weights);

		---- Loop as a workaround to simulate a cursor.
		---- Iterate over the rows in the memory-optimized table  
		---- variable and perform an update for each row.  

		WHILE @m <= @maxM
		BEGIN
			WHILE @n <= @maxN
			BEGIN
				SELECT @gradient_val = CellValue
				FROM @gradient_layer1_weights
				WHERE RowIndex = @m AND ColumnIndex = @n;

				UPDATE dbo.layer1_weights
				SET CellValue = CellValue - @Step * @gradient_val
				WHERE RowIndex = @m AND ColumnIndex = @n;

				SET @n += 1;
			END
			SET @m += 1;
		END
		-- ====================================================================================================================
		
        -- updating weights of biases for Layer 1
		-- save values of gradients for biases in layer 1
		INSERT INTO @gradient_layer1_bias SELECT RowIndex, ColumnIndex, gradient_value FROM
		(SELECT 1 AS RowIndex, ESH.ColumnIndex, sum(ESH.CellValue) AS 
        gradient_value FROM 
        @error_signal_hidden ESH
        GROUP BY ESH.ColumnIndex) as tt6;

		-- NEW CODE FOR UPDATING LAYER1_BIASES 
		-- ================================================================================================================
		SET @m = 1;
		SET @n = 1;  
		SET @maxM = (SELECT max(RowIndex) from dbo.layer1_biases);
		SET @maxN = (SELECT max(ColumnIndex) from dbo.layer1_biases);

		---- Loop as a workaround to simulate a cursor.
		---- Iterate over the rows in the memory-optimized table  
		---- variable and perform an update for each row.  
		WHILE @m <= @maxM
		BEGIN
			WHILE @n <= @maxN
			BEGIN
				SELECT @gradient_val = CellValue
				FROM @gradient_layer1_bias
				WHERE RowIndex = @m AND ColumnIndex = @n;

				UPDATE dbo.layer1_biases
				SET CellValue = CellValue - @Step * @gradient_val
				WHERE RowIndex = @m AND ColumnIndex = @n;

				SET @n += 1;
			END
			SET @m += 1;
		END
		-- ====================================================================================================================
		
	-- while loop ends! 
    END
-- procedure ends 
END
GO

-- The next pieces of code has been used to create a resource pool for a database to use. 
-- For databases that use mem optimized objects we need to create a resource pool, 
-- set the percentage of memory that will be available through it and bind the resource pool to a db. 

-- all mem optimized objects need ~3.5 GB memory (=3324 MB).
-- checking some info about memory used by the memory optimized objects and allocated to them
SELECT * FROM sys.dm_os_sys_info;
SELECT  object_id ,
        OBJECT_SCHEMA_NAME(object_id) + '.' + OBJECT_NAME(object_id) [Table_Name] ,
        memory_allocated_for_table_kb ,
        memory_used_by_table_kb ,
        memory_allocated_for_indexes_kb ,
        memory_used_by_indexes_kb
FROM    sys.dm_db_xtp_table_memory_stats;

-- creating resource pool
CREATE RESOURCE POOL Pool_IMOLTP   
WITH ( MIN_MEMORY_PERCENT = 70, MAX_MEMORY_PERCENT = 70);  
GO  

ALTER RESOURCE GOVERNOR RECONFIGURE;  
GO  
-- binding database to the pool
EXEC sp_xtp_bind_db_resource_pool 'mnist_nn_hekaton', 'Pool_IMOLTP'  
GO
-- message that we get: A binding has been created. Take database 'mnist_nn_hekaton' 
-- offline and then bring it back online to begin using resource pool 'Pool_IMOLTP'.

-- confirming created binding 
SELECT d.database_id, d.name, d.resource_pool_id  
FROM sys.databases d  
GO 

-- now we need to take the database offline and bring it back online
USE master  
GO  

ALTER DATABASE mnist_nn_hekaton SET OFFLINE  
GO  
ALTER DATABASE mnist_nn_hekaton SET ONLINE  
GO  

USE mnist_nn_hekaton  
GO  

-- altering resource pool; increasing the size of max memory percent 
ALTER RESOURCE GOVERNOR DISABLE  

-- change the value of MAX_MEMORY_PERCENT  
ALTER RESOURCE POOL Pool_IMOLTP 
WITH  
     ( MAX_MEMORY_PERCENT = 90 )  
GO  

-- reconfigure the Resource Governor  
--    RECONFIGURE enables resource governor  
ALTER RESOURCE GOVERNOR RECONFIGURE  
GO  
-- it worked now! 

-- if you want to unbind the db from resource pool 
-- unbinding resource pool from a database; 
sys.sp_xtp_unbind_db_resource_pool 'mnist_nn_hekaton';

-- calling TrainNetwork procedure with necessary arguments 
DECLARE @stepss INT
SET @stepss = 3
DECLARE @threshold DECIMAL(10,5)
SET @threshold = 2.2 -- this value can be lower if you want more epochs
EXEC TrainNetwork @threshold, @stepss
GO

-- Forward propagation can be similar to the one in the mnist_mssql file. For checking accuracy on test data
-- we do not need to create memory optimized tables, because we are not interested in optimizing this process. 
