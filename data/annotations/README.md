# Annotations cleaning

`EPIC_100_train.pkl` doesn't cover all IMU videos entirely, and `EPIC_100_validation.pkl` does not cover them at all. Then, we faced a problem with the sampling rate which is not 200Hz at the start or the end. \ 
To solve the previous issues, the following steps has been done:
1. We have filtered out all the videos not covered in a new file called `EPIC_100_train_clean.pkl`
2. From new `EPIC_100_train_clean.pkl`, we filtered out all the actions that are not respecting the sampling rate;
3. We have analyzed how many actions each participant has been done. Based on that:
    1. We have calculated approximately how many actions per each split (train, valid, test) based on length of annotations in 70/10/15% policy.
    2. We have fileld each split **for each participant** as long as the previous criterion has respected;
    3. The remaining participants have been manually reassigned to each split.
 
