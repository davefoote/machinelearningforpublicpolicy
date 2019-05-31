HW5 - Dave Foote

In the walk_through Jupyter Notebook you will find the work I did, the plots I
created, and the ultimate results of my HW5 pipeline, along with comments.

In ml_loop I have placed the functions and helper functions associated with
allowing me to loop through and evaluate different models with different
parameters on different temporal sets, including the data cleaning and temporal
holdouts needed to do so.

In model_analyzer you will find a class I created to store and analyze models
with instances of ModelAnalyzer holding data for a model defined at certain
parameters and hyper parameters and the metrics calculated for that instance.

ml_loop provides code to generate a list of model_analyzer objects and
model_analyzer provides the code to identify the best models from that list.

Execution of that process is found in the walk_through.ipynb file.
