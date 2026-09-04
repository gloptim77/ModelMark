# ModelMark

This tool will help you to test NN models against each other, and form a detailed report that is easy to embed to a website.

**Model evaluation report example:**

<img src="https://github.com/gloptim77/ModelMark/blob/main/result.png" alt="Evaluation result" width="700">

## About:

At each testing run iteration, modelmark:
		
	1) Selects next dataset, input/output len, model and seed 
	2) Seeds the generators for reproducibility
	3) Creates the loader, model and tester objects
	4) Trains the model for E epochs, restores the state with the least validation loss
	5) Tracks the GFLOPs, Memory, Time - all AVG over whole test
	5) Evaluates the model on dataset with metrics from configuration file
	6) Stores the mean result over S runs 

	That way, the more seeds you run, the more "fair" the results are.
	Finally, modelmark will form the report with all the testing results, training stats and your machine metadata.

More on training stats:
		
	Time 		- average time per epoch 
	Params 		- total number of model params
	GFLOPs 		- average per batch
	Peak Memory - max per training iteration

Test consists of F * O * M * S runs, where:
	
	F - number of dataset files in the config (e.g. ["ETTh1" : ..., "Weather" : ...] - means F = 2)
	O - number of input/output sizes (e.g. [32, 64, 128] means O = 3)
	M - number of models (e.g. ["Linear" : ..., "LSTM" : ...] - means M = 2)
	S - number of seeds (e.g. [42, 43, 44] - means S = 2)

# Requirements:
	
	OS: Windows or Linux
	Python: 3.12+

# Usage:

	1) Install the package   
		
		pip install "modelmark @ git+https://github.com/gloptim77/ModelMark.git"

	2) Run the initialization in an empty folder

		modelmark -t init
	
	3.1) It will create two folders "modelmark_files" and "models"
	   
	    In modelmark_files/config.py there are 3 main configs:

	   		model_config = {...} - Models hyperparameters (number of layers, hidden dim, kernel size, etc.)
			data_config = {...} - Dataset parameters (path to file, input/output features, train/val ratios, etc.)
			test_config = {...} - Testing options (optimizer, loss criterion, metrics, learning rate, etc.)

	3.2) (Optional) download the ETT dataset files

		modelmark -t load
 
	4) When your config, model and data are ready, you can run the testing

		modelmark -t run

	5) When the test is over, report results will be in files "result.html" and "result.png"

	Example of the model: "src/modelmark/models/linear.py"
	Example of the config: "src/modelmark/config.py" 

You will find the detailed config example with description at [src/modelmark/config.py](https://github.com/gloptim77/ModelMark/blob/main/src/modelmark/config.py).

Example Linear model with detailed description at [src/modelmark/models/linear.py](https://github.com/gloptim77/ModelMark/blob/main/src/modelmark/models/linear.py).

You can change the config and add your model files to suit your test requirements. 

(But make sure that config and models are compatible)

If you have questions or want to check the source code, go to [ModelMark Github](https://github.com/gloptim77/ModelMark/tree/main).

## Troubleshooting

- **If something doesn't work**

	- Please open an issue and attach your application logs (found at "modelmark_files/modelmark.log") so I can help you troubleshoot. 
	- Try to restart
		```python
		modelmark -t restart
  		```

	- Try to manually delete "modelmark_files" and return to Usage->2)
