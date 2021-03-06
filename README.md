## Recommendation Engines
    
### Recommender of Music Artists
 
Techniques followed

	For as much as we think that personal taste is inexplicable and often hard to describe, recommenders do a
	surprisingly good job of identifying things that match our preferences (books, music tracks and others),
	which we didn’t know we would like.
	
	The algorithm used for the music recommendation engine is the ALS - Alternative Least Squares (collaborative
	filtering), which is indicative for processing large amounts of data in a distributed environment, as it is
	parallelizable and efficient, an important feature for recommender systems to be able to respond quickly in 
	real time applications. It is also already implemented and available as part of the Spark MLLib library.

    	Other techniques applied in recommendation systems around the block are:
        	% content-based filtering recommenders
        	% collaborative-filtering recommendation
        	% classifiers trained on rating-like data
        	% hybrid approaches, which proportially use elements from the above approaches
        	% Graph Neural Networks
	


Data set
   
   	The dataset in this case study is a summarized, sanitized subset of one, released at The 2nd International
	Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011), currently hosted at
	the GroupLens website. The original dataset is available at [1] and its description at [2].
	The sanitized version was drawn from [3]. This is relatively a small dataset expected to fit and be analyzed 
	locally or on one node of your cluster.
	lastfm_artist_list.csv : the artists' dataset, composed of two columns 'artist_id' and 'artist_name' and
				 17493 records					 
	lastfm_user_scrobbles.csv : the user preferences dataset, composed of three columns 'user_id', 'artist_id' 
	                            and 'scrobbles' and of 92792 records. It includes the preferences of 1892 users,
				    92793 scrobble counts and 17493 artists.
    
	

Baseline
	
	As a baseline model is used a simple recommendation system that suggests only the most listened artists to 
	users.	
	
	
	
Challenges
	
	In the case of ALS, the TrainValidationSplit and the cross-validation mechanisms, built in pyspark, cannot 
	be applied, because the algorithm is not able to give predictions for new users, not previously encountered
	in the training set, which makes the algorithm fail as the above mechanisms create random splits in train 
	and validation sets and cannot ensure, during a split, that a user in the train set exists in the validation
	set, too. 
	
	For the model's hyperparameter tuning and evaluation, customized approaches have been used in order to 
	accomodate with the appropriate requirements that the train, dev and test sets should meet in the case of ALS. 	
     
	 

Training process
    
	The model is fitted using the ALS algorithm for an initial set of hyperparameters and after completing 5
	iterations.
	Hyperparameter tuning is performed in a range of values for its hyperparameters on the same train and dev
	sets, without applying cross-validation.
	
   	

Evaluation

	For the evaluation of the recommender system, the data set has been split into train, dev and test set, in
	such a way that each one contains preferences for the same set of users. For each user, ratings are generated
	by the model on what is true preferences for the user, as well as ratings on a set of artists that has been 
	constructed, who are not included in the user's preferences. Susbsequently a score is generated for each user 
	from the fraction of the true preferences that have been rated by the model, higher than the not real preferences.
	This score essentially represents the Area Under the Curve metric in the ROC diagram, expressing the accuracy
	of the model. This approach is described in detail in [4].
	
	A second evaluation approach is also used, deploying a regression evaluator, which returns the root mean square 
	error between the model's ratings and the actual scrobble counts for each artist by user, appropriately normalized
	using the maximum scrobbles count in the data set, so that this newly created column in the dataset expresses the 
	likelihood of listening to an artist and acts as a label in the dataset.
    
    
 
Code

	music_recommender.py
   
   	All can be run interactively with pyspark shell or by submitting e.g. 
       	    exec(open("project/location/recommendation_engines/music_recommender.py").read()) for an all at once 
	execution.
	The code has been tested on a Spark standalone cluster. For the Spark setting, spark-3.1.3-bin-hadoop2.7 
	bundle has been used. The external python packages that are used in this implementation exist in the 
	requirements.txt file. Install with: 
		pip install -r project/location/recommendation_engines/requirements.txt
	This use case is inspired from the series of experiments presented in [4], though it deviates from it, in
	the programming language, the setting used and in the analysis followed.
   


References

	1. https://grouplens.org/datasets/hetrec-2011/
	2. https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-readme.txt
	3. https://www.kaggle.com/pcbreviglieri/lastfm-music-artist-scrobbles
	4. Advanced Analytics with Spark, Sandy Ryza, Uri Laserson, Sean Owen, & Josh Wills
	
