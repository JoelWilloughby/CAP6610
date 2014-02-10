import sys
import re
import csv
import random
import math

class Naive_Bayes:
    def __init__(self, training_dict, classes, training, bins, n_bins, data):
        self.training_dict = training_dict
        self.training = training
        self.classes = classes
        self.bins = bins
        self.data = data
        self.n_bins = n_bins
        self.n_attribs = len(bins)
        self.n_classes = len(training_dict)
        self.class_totals = []
        for cl in classes:
            self.class_totals.append(len(training_dict[cl]))
        self.training_size = sum(self.class_totals)

    def cond_prob(self, feature, value, cl):
        b = int((value-self.bins[feature][0])/self.bins[feature][1])
        if b >= self.n_bins: b = self.n_bins-1
        if b < 0: b = 0
        if self.class_totals[cl] == 0: return 0
        return float(self.data[cl][feature][b])/float(self.class_totals[cl])

    def class_prob(self, test, cl):
        prob = float(self.class_totals[cl])/self.training_size
        for i in range(0, self.n_attribs):
            prob *= self.cond_prob(i, test[i], cl)
        return prob

    def label(self, test):
        probs = []
        test_bins = []
        i = 0
        for i in range(0, self.n_classes):
            prob = self.class_prob(test, i)
            probs.append(prob)
        return (self.classes[probs.index(max(probs))], max(probs))

    def get_statistics(self):
        tabs = ' ' * 3
        string  = 'Training: ' + str(self.training_size) + '\n'
        string += '  '
        for cl in self.classes:
            string += ' ' + cl + (': %.2f' % (float(len(self.training_dict[cl]))/self.training_size)) + ','
        string += '\nBins: ' + str(self.n_bins) + '\n'
        string += '  '
        for bin in self.bins:
            string += ' ' + '({0:.2f}, {1:.2f}),'.format(bin[0], bin[1])
        return string


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
    return random.sample(data, n)

def find_classes(data):
    classes = []
    for dat in data:
        a, s = dat
        if not s in classes:
            classes.append(s)
    return classes 

def make_class_dict(training, classes):
    # attribs = len(training[0])
    num_classes = len(classes)
    empty = []
    for i in range(0, num_classes):
        empty.append([])
    temp = dict(zip(classes, empty))
    for c in classes:
        for el in training:
            a, s = el
            if s == c:
                temp[c].append(a)
    return temp

def make_naive_bayes(training, classes, bins):
    b_dic = make_class_dict(training, classes)
    if(bins == 0):
        # n_bins = int(1 + math.log(len(training), 2))
        n_bins = 3
    else:
        n_bins = bins
    bins = []
    bayes = []
    n_attribs = len(training[0][0])
    for i in range(0, n_attribs):
        small = 1e20
        big = -1e20
        for el in training:
            if el[0][i] < small:
                small = el[0][i]
            if el[0][i] > big:
                big = el[0][i]
        bin_length = (big-small)/n_bins
        bins.append((small, bin_length))
    for cla in classes:
        value = b_dic[cla]
        temp = []
        for i in range(0, n_attribs):
            temp2 = [0]*n_bins
            for cl in value:
                b = int((cl[i]-bins[i][0])/bins[i][1])
                if b == n_bins: b = b -1
                temp2[b] = temp2[b] + 1
            temp.append(tuple(temp2))
        bayes.append(tuple(temp))
    return Naive_Bayes(b_dic, classes, training, bins, n_bins, bayes)

def init_naive(file_name, percent, bins):
    data = parse_data(read_data(file_name))
    classes = find_classes(data)
    training_set = pick_training_set(data, percent)
    bayes = make_naive_bayes(training_set, classes, bins)
    for t in training_set:
        data.remove(t)
    return (data, classes, bayes)

def run_test(bayes, test_data):
    incorrect = 0
    for test in test_data:
        if bayes.label(test[0])[0] != test[1]:
            incorrect += 1

    return float(incorrect)/len(test_data)

def test_set(n, filename, percent, bins):
    total = 0.0
    for i in range(0,n):
        data, classes, bayes = init_naive(filename, percent, bins)
        print bayes.get_statistics()
        temp = run_test(bayes, data)
        print 'Error Percent: ' + '{0:.2f}'.format(temp)
        total += temp
        print '\n' + ('=' * 15) + '\n'
    print 'AVERAGE = ' + '{0:.3f}'.format(total/n)

def main():
    stuffs = parse_data(read_data('iris.data'))
    classes = find_classes(stuffs)
    new_stuffs = pick_training_set(stuffs, 0.50)
    for stuff in new_stuffs:
      print stuff
    print classes
    bayes = make_naive_bayes(new_stuffs, classes)
    print bayes.training_dict
    print bayes.classes
    count = 0
    for thing in stuffs:
        result = bayes.label(thing[0])
        if result[0] == thing[1]: count = count + 1
        print str(thing[1]) + str(bayes.label(thing[0]))

    print str(count - bayes.training_size) + '/' + str(150 - bayes.training_size) 
    print float(count - bayes.training_size)/(150-bayes.training_size)

if __name__ == '__main__':
    main()