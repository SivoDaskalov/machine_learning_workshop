from tutoring.wrangling import assemble
from tutoring.regression import simple_linear_regression, enet_model_selection, enet_cv_model_selection
from tutoring.classification import *

X, y = assemble("boston")
model = simple_linear_regression(X, y, plot_figures = False)
model = enet_model_selection(X, y)
model = enet_cv_model_selection(X, y)

X, y = assemble("breast_cancer")
model = train_and_eval_knn(X, y)
model = batch_train_models(X, y)

pass