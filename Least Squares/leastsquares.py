import sys
import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def read_data(file):
    data = []
    labels = []
    f = open(file, 'rb')
    return csv.reader(f)

def parse_data(reader):
    data = []
    labels = []
    for row in reader:
        data.append(map(float, row[0:4]))
        labels.append(row[4])
    return zip(data, labels)

def pick_training_set(data, percent):
    n = int(percent*len(data))
    a = random.sample(data, n)
    for t in a:
        if t in data:
            data.remove(t)
    return (data,a)

def get_matrices(data, classes):
    n_attribs = len(data[0][0])
    xs = []
    ts = []
    n_classes = len(classes)
    for (x,c) in data:
        i = classes.index(c)
        xs.append([1.0]+x)
        temp = [0.0] * n_classes
        temp[i] = 1.0
        ts.append(temp)
    return (np.matrix(xs), np.matrix(ts))

def make_least_squares(xs, ts, lamb):
    # print lamb
    w = np.transpose(xs) * xs
    # print w
    w = w + lamb * np.identity(len(xs.tolist()[0]))
    # print w
    w = np.linalg.inv(w)
    w = w * np.transpose(xs)
    w = w * ts
    # print w
    return w

class Least_Squares:
    def __init__(self, filename, percent, lamb):
        self.data = parse_data(read_data(filename))
        self.classes = []
        for dat in self.data:
            if dat[1] not in self.classes:
                self.classes.append(dat[1])
        self.test_data, self.train_data = pick_training_set(self.data, percent)
        self.train_mats, self.train_label_mats = get_matrices(self.train_data, self.classes)
        self.test_mats, self.test_label_mats = get_matrices(self.test_data, self.classes)
        self.w_mat = make_least_squares(self.train_mats, self.train_label_mats, lamb)

    def test(self, x):
        res = x * self.w_mat
        res_list = res.tolist()[0]
        test_class = res_list.index(max(res_list))
        return (test_class, self.classes[test_class])

    def get_eig_vals(self):
        return np.linalg.eigvals(np.transpose(self.w_mat) * self.w_mat).tolist()

    def tweak_lambda(self, lam):
        self.w_mat = make_least_squares(self.train_mats, self.train_label_mats, lam)

def test_all(lesq):
    # print 'TRAINING'
    correct = 0
    total = 0
    for i in range(0,len(lesq.train_mats)):
        tm = lesq.train_mats[i]
        tl = lesq.train_label_mats[i].tolist()[0]
        res_index, res_class = lesq.test(tm)
        if res_index is tl.index(max(tl)):
            correct += 1
        # print tm
        # print '   ' + str(tl) + ' ' + str(res_index)
        total += 1
    # print str(correct) + '/' + str(total)
    # print float(total-correct)/total
    tr = float(total-correct)/total

    # print '\nTESTING'
    correct = 0
    total = 0
    for i in range(0,len(lesq.test_mats)):
        tm = lesq.test_mats[i]
        tl = lesq.test_label_mats[i].tolist()[0]
        res_index, res_class = lesq.test(tm)
        if res_index is tl.index(max(tl)):
            correct += 1
        # print tm
        # print '   ' + str(tl) + ' ' + str(res_index)
        total += 1
    # print str(correct) + '/' + str(total)
    # print float(total-correct)/total
    te = float(total-correct)/total
    return (tr, te)

def run_some_tests(percent, lams, n):
    trs = [0.0] * len(lams)
    tes = [0.0] * len(lams)
    for i in range(0,n):
        lesq = Least_Squares('iris.data', percent, 0)
        for lam in lams:
            lesq.tweak_lambda(lam)
            train, test = test_all(lesq)
            trs[lams.index(lam)] += train
            tes[lams.index(lam)] += test

    for i in range(0, len(lams)):
        print 'LAMBDA = ' + str(lams[i])
        print '   TRAINING  ' + str(trs[i]/n)
        print '   TESTING   ' + str(tes[i]/n) + '\n'
    return [t/n for t in trs], [t/n for t in tes]

def plot_results(res_lam, res_train, res_test):
    plt.figure(1)
    plt.subplot(211)
    plt.title("TRAINING")
    plt.plot(res_lam, res_train, 'bo')

    plt.subplot(212)
    plt.title("TESTING")
    plt.plot(res_lam, res_test, 'ro')
    plt.show()
