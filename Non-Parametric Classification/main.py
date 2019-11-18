import csv
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASS = 10

train_data = dict()
test_data = dict()

with open("./data/train1k.csv", "rt") as file:
    lines = csv.reader(file, delimiter=",")
    next(lines)
    for line in lines:
        key = int(line[0])
        value = list(map(int, line[1:]))
        if key not in train_data.keys():
            train_data[key] = list()
        train_data[key].append(value)

with open("./data/test1k.csv", "rt") as file:
    lines = csv.reader(file, delimiter=",")
    next(lines)
    for line in lines:
        key = int(line[0])
        value = list(map(int, line[1:]))
        if key not in test_data.keys():
            test_data[key] = list()
        test_data[key].append(value)

# print(train_data[1])
# print(len(test_data))

train_means = dict()
test_means = dict()

for key, value in train_data.items():
    train_means[key] = list(map(lambda x: x // len(value), list(map(sum, zip(*value)))))
    # print(key, train_means[key])

for key, value in test_data.items():
    test_means[key] = list(map(lambda x: x // len(test_data), list(map(sum, zip(*value)))))


def im_show(img):
    img = np.reshape(img, (28, 28))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


for key, value in train_means.items():
    im_show(value)


def calc_distance(value_, mean_):
    distance = [(v_ - m_)**2 for v_, m_ in zip(value_, mean_)]
    return sum(distance)**0.5


nearest_mean_results = list()
train_data_gt = list()

for l in range(NUM_CLASS):
    for v in train_data[l]:
        train_data_gt.append(l)
        distances = list()
        for i in range(NUM_CLASS):
            distances.append(calc_distance(v, train_means[i]))
        nearest_mean_results.append(distances.index(min(distances)))

# print(train_data_gt)
# print(nearest_mean_results)


def compute_confusion_matrix(y_true, y_pred):
    K = len(np.unique(y_true))
    result = np.zeros((K, K))

    for i in range(len(y_true)):
        result[y_true[i]][y_pred[i]] += 1

    return result


train_cm = compute_confusion_matrix(train_data_gt, nearest_mean_results)
print("Confusion matrix for training data: \n{}".format(train_cm))
plt.matshow(train_cm)
plt.colorbar()
# plt.savefig('./images/training_cm.png')
plt.show()

test_data_gt = list()
for l in range(NUM_CLASS):
    for v in test_data[l]:
        test_data_gt.append(l)

test_cm = compute_confusion_matrix(test_data_gt, nearest_mean_results)
print("Confusion matrix for test data: \n{}".format(test_cm))
plt.matshow(test_cm)
plt.colorbar()
# plt.savefig('./images/test_cm.png')
plt.show()


def KNN(k, set1, set2):
    results = list()
    for key1, value1 in set1.items():
        for v1 in value1:
            distances = list()
            for key2, value2 in set2.items():
                for v2 in value2:
                    distances.append((key1, key2, calc_distance(v1, v2)))

            distances.sort(key=lambda x: x[2])
            distances = distances[1:k+1]  # reduce the sample itself

            preds = [k2 for k1, k2, d in distances]
            results.append(max(set(preds), key=preds.count))
    return results


_1nn_results = KNN(1, test_data, test_data)

test_cm_1nn = compute_confusion_matrix(test_data_gt, _1nn_results)
print("Confusion matrix for test data: \n{}".format(test_cm_1nn))
plt.matshow(test_cm_1nn)
plt.colorbar()
plt.savefig('./images/test_cm_1nn.png')
plt.show()


def evaluate(y_true, y_pred):
    tp = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp += 1
    accuracy = tp / len(y_true)
    print("Accuracy: {}%".format(accuracy * 100))
    return accuracy * 100


acc_1nn = evaluate(test_data_gt, KNN(1, train_data, test_data))
acc_2nn = evaluate(test_data_gt, KNN(2, train_data, test_data))
acc_3nn = evaluate(test_data_gt, KNN(3, train_data, test_data))
acc_4nn = evaluate(test_data_gt, KNN(4, train_data, test_data))
acc_5nn = evaluate(test_data_gt, KNN(5, train_data, test_data))
acc_6nn = evaluate(test_data_gt, KNN(6, train_data, test_data))
acc_7nn = evaluate(test_data_gt, KNN(7, train_data, test_data))
acc_8nn = evaluate(test_data_gt, KNN(8, train_data, test_data))
acc_9nn = evaluate(test_data_gt, KNN(9, train_data, test_data))
acc_10nn = evaluate(test_data_gt, KNN(10, train_data, test_data))


