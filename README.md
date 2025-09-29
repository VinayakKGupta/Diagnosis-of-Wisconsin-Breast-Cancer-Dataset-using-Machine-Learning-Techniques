## Key findings - Breast Cancer Dataset Analysis ##

These projects were part of my university coursework and involved extensive discussions with peers and subject coordinators. I highly recommend reviewing the full report for a comprehensive analysis. 

In addition to the descriptive and diagnostic work done Python libraries such as Scikit-learn, NumPy, and Pandas were also employed to carry out advanced analyses. Techniques including voting classifiers, logistic regression, SVMs, and random forests were applied to extract predictive insights and strengthen the data-driven recommendations.

**Methodology - Reasoning Behind the Approach**

The analysis used the Breast Cancer dataset and followed a rigorous pipeline:

Data preprocessing: The id column was dropped (non-informative), and diagnosis labels were mapped to 1 (malignant) and 0 (benign). Stratified sampling was applied because the dataset is imbalanced (~35% malignant, 65% benign).

Exploratory analysis: Identified feature skew, class imbalance, and high correlations among size-related features (e.g., radius, area, perimeter). This highlighted the need for dimensionality reduction or regularisation.

Model choice:

SVM (RBF kernel) was tuned with GridSearchCV using 5-fold Stratified Cross-Validation to preserve the class ratio across folds.

Recall (sensitivity) was optimized, not accuracy. The reasoning: in cancer diagnosis, false negatives (missed cancers) are far costlier than false positives.

The hyperparameter grid covered a range of C (0.1, 1, 10) and gamma (0.01, 0.1, 1), giving 9 combinations × 5 folds = 45 models.

This strategy ensured robustness, avoided bias from class imbalance, and balanced under- vs. over-regularization.

<img width="590" height="485" alt="image" src="https://github.com/user-attachments/assets/2f7e791a-974c-4dce-9cee-e7555f4d24fc" />

<img width="1281" height="235" alt="image" src="https://github.com/user-attachments/assets/c957fdff-7531-43c7-9f38-43885d33dea8" />

Results – Key Findings

Best SVM model: C = 1, gamma = 0.1, giving a cross-validated recall of ~0.959 (≈96%).

Impact of class imbalance: Optimizing accuracy would have been misleading since always predicting “benign” yields ~65% accuracy. Focusing on recall ensured malignant cases were detected.

Bias–variance trade-off: Higher regularisation (large α in ridge/SGD) flattened decision boundaries, leading to underfitting; low regularisation captured patterns better but risked variance.

Closed-form vs. SGD: For small/medium regularisation, SGDRegressor (with scaling) matched the closed-form Ridge Regression very closely (MSE differences ≈0.01). At extreme values (α=100, higher-degree polynomials), SGD underfit and became unstable, while the closed-form solution remained numerically stable.

Efficiency: Closed-form solutions were 3–16× faster to fit and more stable compared to SGD on polynomial features up to degree 10, though SGD would scale better for massive datasets.

<img width="620" height="473" alt="image" src="https://github.com/user-attachments/assets/fb2a8cc1-3001-481b-92c5-778485169d2d" />


The methodology was guided by clinical reasoning (recall > accuracy) and data characteristics (imbalance, collinearity). The results showed that careful tuning of SVM hyperparameters achieved very high recall, while comparisons of closed-form vs. SGD highlighted the importance of feature scaling and appropriate regularisation.




