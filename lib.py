import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def plot_accuracy(history,epochs):
    plt.figure(figsize=(15, 7))
    plt.plot(range(epochs), history.history['accuracy'])
    plt.plot(range(epochs), history.history['val_accuracy'])
    plt.legend(['training_acc', 'validation_acc'])
    plt.title('Accuracy')

def train_test_accuracy(model,X_train,Y_train,X_test,Y_test):
    train_results = model.evaluate(X_train, np.asarray(Y_train), verbose=0, batch_size=256)
    test_results = model.evaluate(X_test, np.asarray(Y_test), verbose=0, batch_size=256)
    print(f'Train accuracy: {train_results[1] * 100:0.2f}')
    print(f'Test accuracy: {test_results[1] * 100:0.2f}')



def report_nn(model,X_test,Y_test):
    model = model
    y_actuals = np.argmax(Y_test, axis=1)

       # encoder.inverse_transform(a)
    y_preds = model.predict(X_test)  # model.predict([val_X], batch_size=1024, verbose=0)

    prediction_ = np.argmax(y_preds, axis=1)

    y_preds = prediction_
    target_names = ['0', '1', '2', '3', '4', '5']
    # print(recall_score(y_actuals,prediction_,average='macro'))
    report = classification_report(y_actuals.tolist(), y_preds.tolist(), target_names=target_names)

    print(report)

    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_actuals.tolist(), y_preds.tolist())))


def print_confusion(model,X_test,Y_test):
    model = model
    a = np.argmax(Y_test, axis=1)
    # print(a)
    y_actuals = a  # encoder.inverse_transform(a)
    y_preds = model.predict(X_test)  # model.predict([val_X], batch_size=1024, verbose=0)

    # corr = d.corr()

    # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=bool))

    prediction_ = np.argmax(y_preds, axis=1)
    # print('prediction',prediction_)
    # print('--Confusion Matrix for ',)
    # print('/n')
    cm = confusion_matrix(y_actuals, prediction_)
    cm_df = pd.DataFrame(cm,
                         index=['0', '1', '2', '3', '4', '5'],
                         columns=['0', '1', '2', '3', '4', '5'])
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=2)
    sns.heatmap(cm_df, annot=True, cmap='Blues')  # ,mask=mask)
    # plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels', fontsize=30, weight='bold')
    plt.xlabel('Predicted Labels', fontsize=30, weight='bold')
    plt.savefig("conf_mat.pdf", dpi=1000)

    plt.show()






