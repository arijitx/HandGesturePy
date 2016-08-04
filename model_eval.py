import svm_train as st
from sklearn.svm import SVC
import numpy as np
from sklearn import svm, cross_validation

n_class=input("Number of Class : ")
n_item_per_calss=input("Number of Item per Class : ")
folder_name=raw_input("Folder Name : ")

print 'Loading Dataset . . .'
imgs,labels=st.load_img_labels(n_class,n_item_per_calss,folder_name)
samples=st.preprocess_hog(imgs)

n_cv=input("Number of Cross validation Folds : ")

scoring =['accuracy', 'average_precision', 'f1_macro', 'f1_micro', 'f1_samples', 
		 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 
		 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

print 'Using Scoring Strategies ',scoring

clf=SVC()

for scr in scoring:
	print '----------------------- '+scr+' -----------------------'
	try:
		res=cross_validation.cross_val_score(clf, samples, labels, scoring=scr,cv=n_cv)
		print 'Mean    :   ',res.mean()
		print 'Max     :   ',res.max()
		print 'Min     :   ',res.min()
	except ValueError:
		print 'Multiclass not Supported '