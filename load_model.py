from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import pickle
import skvideo.io
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time

# tensorboard --logdir=logs --host=127.0.0.1
model = load_model('models/model-8-16-16-d256')
CATEGORIES = ["About","Absolutely","Abuse","Access","According","Accused","Across","Action","Actually",
              "Allow","Today","Tomorrow","Yesterday"]

#NAME = "More-Words-8-16-16-d64-1554732208"

#tensor_b = TensorBoard(log_dir='logs/{}'.format(NAME))

# load data
x_test = pickle.load(open("data/data_test1.pickle", "rb"))
y_test = pickle.load(open("data/label_test.pickle", "rb"))

# normalize to 0-1
x_test = x_test/255.0

errors = np.zeros((13,13))

count = 0
y_pred = np.zeros(y_test.shape)

# model.fit(x_train, y_train, batch_size=128, validation_data=(x_val, y_val), epochs=5, callbacks=[tensor_b])
for i in range(614):
    x = x_test[i].reshape((1,29,32,32,1))
    label = model.predict_classes(x)
    y_pred[i] = label
    if label != y_test[i]:
        count += 1
        classified = CATEGORIES[label[0]]
        correct = CATEGORIES[y_test[i]]
        print("Video id: {:<10} Classified label: {:<15} Correct label: {:<15}".format(i, classified, correct))
        errors[label[0]][y_test[i]] += 1
        skvideo.io.vwrite("errors/Error_{}_{}.gif".format(classified, correct), x_test[i] * 255)

print("{:<20} {:<20} {:<20}".format("Number", "Correct Label", "Predicted"))

for i in range(13):
    for j in range(13):
        if errors[j][i] > 0:
            print("{:<20} {:<20} {:<20}".format(int(errors[j][i]), CATEGORIES[i], CATEGORIES[j]))

print(count)

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=CATEGORIES, yticklabels=CATEGORIES,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=CATEGORIES,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=CATEGORIES, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



# model.save("models/model4-8-16-16-d64")