import pydotplus
from six import StringIO
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif


# Question 2(a)
def load_data(real: str, fake: str):
    real = open(real, "r")
    fake = open(fake, "r")
    real_lst = []
    fake_lst = []
    real_label = []
    fake_label = []
    for l in real.readlines():
        real_lst.append(l.strip())
        real_label.append(1)
    for l in fake.readlines():
        fake_lst.append(l.strip())
        fake_label.append(2)
    real_fake_lst = real_lst + fake_lst
    real_fake_label = real_label + fake_label

    cv = CountVectorizer()
    cv_trans = cv.fit_transform(real_fake_lst).toarray()
    names = cv.get_feature_names()

    train_data, other_data, train_label, other_label = train_test_split(cv_trans, real_fake_label, test_size=0.3)
    test_data = other_data[:len(other_data) // 2]
    test_label = other_label[:len(other_data) // 2]
    valid_data = other_data[len(other_data) // 2:]
    valid_label = other_label[len(other_data) // 2:]

    return train_data, train_label, test_data, test_label, valid_data, valid_label, names, cv_trans


# Question 2(b)
def select_model(train_data, valid_data, train_label, valid_label, depth_list):
    lst = []
    for d in depth_list:
        print("Depth:", d)
        gini = DecisionTreeClassifier(criterion="gini", max_depth=d)
        gini = gini.fit(train_data, train_label)
        gini_predict = gini.predict(valid_data)
        gini_accuracy = metrics.accuracy_score(valid_label, gini_predict)
        print("Gini Accuracy:", gini_accuracy)

        entropy = DecisionTreeClassifier(criterion="entropy", max_depth=d)
        entropy.fit(train_data, train_label)
        entropy_predict = entropy.predict(valid_data)
        entropy_accuracy = metrics.accuracy_score(valid_label, entropy_predict)
        print("Information Gain Accuracy:", entropy_accuracy)

        data = [gini, gini_accuracy, entropy, entropy_accuracy]
        lst.append(data)
    return lst


# Question 2(d)
def compute_information_gain(real, fake, keyword):
    train_data, train_label, test_data, test_label, valid_data, valid_label, names, cv_trans = \
        load_data("clean_real.txt", "clean_fake.txt")
    res = dict(zip(names, mutual_info_classif(train_data, train_label, discrete_features=True)))

    # print(res)
    print("Keyword chosen for the split:", keyword)
    print("Information Gain on respective keyword:", res[keyword])

    return keyword, res[keyword]


if __name__ == "__main__":
    # Question 2(a) and 2(b)
    train_data, train_label, test_data, test_label, valid_data, valid_label, names, cv_trans = \
        load_data("clean_real.txt", "clean_fake.txt")
    select_model(train_data, valid_data, train_label, valid_label, [5, 10, 20, 40, 80])

    # Question 2(c)
    dot_data = StringIO()
    highest_accuracy_model = DecisionTreeClassifier(criterion="entropy", max_depth=80)
    highest_accuracy_model = highest_accuracy_model.fit(train_data, train_label)
    export_graphviz(highest_accuracy_model, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=names, class_names=['real', 'fake'], max_depth=2)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('visualization.png')

    # Question 2(d)
    compute_information_gain("clean_real.txt", "clean_fake.txt", "donald")
    compute_information_gain("clean_real.txt", "clean_fake.txt", "trumps")
    compute_information_gain("clean_real.txt", "clean_fake.txt", "the")
    compute_information_gain("clean_real.txt", "clean_fake.txt", "hillary")
    compute_information_gain("clean_real.txt", "clean_fake.txt", "destroyed")
    compute_information_gain("clean_real.txt", "clean_fake.txt", "trump")
