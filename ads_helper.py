import sys
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


tdfn = pd.read_csv("known_identity_df.csv")

mention_threshold = .1



vectorizer = TfidfVectorizer(stop_words='english')
text_mat = vectorizer.fit_transform(tdfn.comment_text)
lr = LogisticRegression()
clf = lr.fit(text_mat,tdfn.predicted_toxicity)

model=clf
Y_test = tdfn.predicted_toxicity
# Get the probability of Y_test records being = 1
Y_test_probability_1 = lr.predict_proba(text_mat)[:, 1]

# Use the metrics.roc_curve function to get the true positive rate (tpr) and false positive rate (fpr)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_probability_1)

# Plot the ROC curve
plt.plot(fpr, tpr)



#sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')

c = make_pipeline(vectorizer,clf)
clf_pred = c.predict(tdfn.comment_text)
tdfn["clf_pred"] = clf_pred



def get_incorrectly_tagged_indices(df,subgroup,mention_threshold,incorrect_tag):
    df_sub = df[df[subgroup] > mention_threshold]
    df_sub = df_sub[df_sub["bin_toxicity"] == incorrect_tag]
    df_sub_clf = df_sub[df_sub["predicted_toxicity"] == df_sub["clf_pred"]]
    return df_sub_clf[df_sub_clf["bin_toxicity"] != df_sub_clf["predicted_toxicity"]].index.values



def interpret_data(func,test_set,target, class_names,indices=range(5),num_indices = 30,plot=False,file_prefix="part_b_i"):
    explainer = LimeTextExplainer(class_names=class_names)
    word_lists = []
    if len(indices) > num_indices:
        indices = indices[:num_indices]
    for r_idx in indices:
        exp = explainer.explain_instance(test_set[r_idx], func, num_features=10)
        if plot:
            #fig = exp.as_pyplot_figure()
            if target[r_idx] == 1:
                tclass = "Toxic"
            else:
                tclass = "Non-Toxic"
            print(F"the correct classification is {tclass}")
            exp.show_in_notebook()
            #exp.save_to_file(f"{file_prefix}_{r_idx}.html")
        word_lists.append(dict(exp.as_list()))
        
    d = {}
    for wd in word_lists:
        for k,v in wd.items():
            if len(k) < 2:
                continue
            if k not in d:
                d[k] = abs(wd[k])
            else:
                d[k] += abs(wd[k])
        
    return d


def get_incorrect_examples(tdfn,identity,mention_threshold,toxicity):
    get_incorrect_examples.c = make_pipeline(vectorizer, clf)
    indices =get_incorrectly_tagged_indices(tdfn,identity,mention_threshold,toxicity)
    return interpret_data(get_incorrect_examples.c.predict_proba, tdfn.comment_text,tdfn["bin_toxicity"],['non_toxic','toxic'],indices = indices[0:5])



