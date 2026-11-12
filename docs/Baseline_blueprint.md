## Baseline blueprint

### We intend to use a sliding window approach producing one feature vector per child:
Here we record one (20 x 57,664) sample
- drop the zero-padded tail
- we cut into segmented windows, such as 30 seconds, which is about 30 per child
- features per window and channel, (Hjorth with band powers from Welch's method with 4 second segments)
- then take the median over the windows which translates to one feature vector per child (we take the mean for articfact protection)

X_train is then: 825 x n_features and y_train: 825 ages
- sklearn pipeline to fit on train (standardscaler, Ridge/RF)
- then choose features and hyperparams by validation mean absolute error
- then we touch the test split just once at the end

### How this relates to our own CNN that we will make later:
With both approaches using the median evaluation method, both approaches are more similar and it makes the comparison cleaner, the downside of this would be that rows from one child wouldn't be independent, so cross-validation must keep each child in one group or fold before validating.
- **window length:** since just one second gives a noisy 1Hz frequency resolution, lets try 10-30s 
- **light sleep vs N2 sleep:** we need to compute the features separately for 0-300s (before N2  sleep) and 300-900s (N2 sleep), then use both sets. According to the dataset paper the pre-N2 part is more accurate, so confirming this would be a good check early.
- **model choice:** In the age range 0-3, amplitude barely changes then it drops rapidly. A sraight line in raw age wont be able to fit that. So we can either predict log(age) with ridge regression or use tree ensembles in some way? this would be able to handle the non-linear relationships. For now lets use ridge on log features predicting log(age) as a first baseline


### The class imbalance (way more younger people in the dataset):
regressors have no so called "majority classes" to fall back on, but the imbalance still does happen in a weaker form:
- error is lowest where the dataset in densest, so toddlers
- sparse ages just get pulled towards the middle, teenagers get predicted too young and the youngest infants get predicted too old, this is regression to the mean. Tree models are especially prone to it (avg over leaves, with only 77 teenagers in training those leaves mix in with the younger children)
- Plain mean square error actually hides it, since half the test children are under 3.
- the imbalance is in the years though, not in the log(age), children aged 1 and up are spread almost evenly, the EEG changes fastest in the youngest ages, by training on log(age) we can largely remove the imbalance automatically.