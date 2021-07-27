import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import ImageTk, Image
from gensim.models import Word2Vec
from keras import layers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import predictText as p

plt.style.use('ggplot')

df = pd.read_csv('cleaned_fix.csv', encoding='latin')
df.narasi = df.narasi.astype(str)
X = df

df.narasi = df.narasi.str.split()

sentences_train, sentences_test, y_train, y_test = \
    train_test_split(df.narasi, df.label, test_size=0.3, random_state=100)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_train)
seq_train = tokenizer.texts_to_sequences(sentences_train)
seq_test = tokenizer.texts_to_sequences(sentences_test)

encoded_tfidf_train = tokenizer.sequences_to_matrix(seq_train, mode="tfidf")
encoded_tfidf_test = tokenizer.sequences_to_matrix(seq_test, mode="tfidf")
X_train = pad_sequences(encoded_tfidf_train, dtype='float64', padding="post", truncating="post")
X_test = pad_sequences(encoded_tfidf_test, dtype='float64', padding="post", truncating="post")
EMBEDDING_DIM = 1000
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(layers.Embedding(vocab_size, 100, input_length=vocab_size))
model.add(layers.Conv1D(16, 5, activation="relu"))
model.add(layers.MaxPooling1D())
model.add(layers.Conv1D(32, 5, activation="relu"))
model.add(layers.MaxPooling1D())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                    epochs=5, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
loss, accuracy = model.evaluate(X_train, y_train)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test)
print("Testing Accuracy:  {:.4f}".format(accuracy))

y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

cf_matrix = confusion_matrix(y_test, y_pred.round())
print(confusion_matrix(y_test, y_pred.round(), labels=[1, 0]))
print(classification_report(y_test, y_pred.round()))

sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
x = range(1, len(acc) + 1)

plot = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, acc, 'b', label='Training acc')
plt.plot(x, val_acc, 'r', label='Validation acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, loss, 'b', label='Training loss')
plt.plot(x, val_loss, 'r', label='Validation loss')
plt.title('Loss')
plt.legend()
plt.show()

instance = X.narasi[19]
print(instance)

instance = tokenizer.texts_to_sequences(instance)
flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)
flat_list = [flat_list]
encoded_tfidf = tokenizer.sequences_to_matrix(flat_list, mode="tfidf")
instance = pad_sequences(encoded_tfidf, dtype='float64', padding="post", truncating="post")

predict = model.predict(instance)
if predict < 0.5:
    print('Asli')
else:
    print('Palsu')


def helloCallBack():
    entryText.set(" ")
    instance = judulEntry.get("1.0", 'end-1c') + " " + narasiEntry.get("1.0", 'end-1c')
    print(instance)
    a = p.clean(instance)

    instance = tokenizer.texts_to_sequences(instance)
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)
    flat_list = [flat_list]
    encoded_tfidf = tokenizer.sequences_to_matrix(flat_list, mode="tfidf")
    instance = pad_sequences(encoded_tfidf, dtype='float64', padding="post", truncating="post")

    predict = model.predict(instance)
    if predict < 0.5:
        print('Asli')
        hasil = 'Asli'
    else:
        print('Palsu')
        hasil = 'Palsu'

    entryPre.set(hasil)


def clearText():
    judulEntry.delete("1.0", 'end-1c')
    narasiEntry.delete("1.0", 'end-1c')
    # preEntry.set("")


root = tk.Tk()
root.geometry("1200x800")

titleFrame = tk.Frame(root, height=100)
titleFrame.pack_propagate(0)
titleFrame.pack(fill='both', side='top')

title = tk.Label(titleFrame, text="PREDIKSI BERITA PALSU MENGGUNAKAN METODE \n CONVOLUTION NEURAL NETWORK",
                 font=('Calibri', 14))
title.pack(fill='both', expand='True')
nama = tk.Label(titleFrame, text="Hieronimus Fredy Morgan / 175314080", font=('Calibri', 12))
nama.pack(pady=10, fill='both', expand='True')

accuracyFrame = tk.Frame(root)
accuracyFrame.pack_propagate(0)
accuracyFrame.pack(fill='both', side='left', expand='True')

accuracyLabel = tk.Label(accuracyFrame, text="ACCURACY", font=('Calibri', 14))
accuracyLabel.pack(fill='both', side='top', pady=20)

entryText = tk.StringVar()

accuracyEntry = tk.Entry(accuracyFrame, state='disabled', textvariable=entryText, justify='center',
                         font=('Calibri', 14))
entryText.set(accuracy)
accuracyEntry.pack(ipadx=20, ipady=10, pady=20)

img = ImageTk.PhotoImage(Image.open("myplot.png").resize((300, 300), Image.ANTIALIAS))
img2 = ImageTk.PhotoImage(Image.open("without confusion.png").resize((300, 300), Image.ANTIALIAS))
#
# #
# canvas = tk.Canvas(accuracyFrame, width=300, height=300)
# # img = ImageTk.PhotoImage(Image.open("word2vec.png"))
# canvas.create_image(150, 150, anchor='center', image=img)
# canvas.pack(side='left')

# plotting the graph
canvas = FigureCanvasTkAgg(plot, master=accuracyFrame)
canvas.draw()

# placing the canvas on the Tkinter window
canvas.get_tk_widget().pack()

# creating the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas,
                               accuracyFrame)
toolbar.update()

# placing the toolbar on the Tkinter window
canvas.get_tk_widget().pack()
canvas1 = tk.Canvas(accuracyFrame, width=300, height=300)
# img = ImageTk.PhotoImage(Image.open("word2vec.png"))
canvas1.create_image(150, 150, anchor='center', image=img2)
canvas1.pack(side='right')

predictFrame = tk.Frame(root)
predictFrame.pack_propagate(0)
predictFrame.pack(fill='both', side='right', expand='True')

predictLabel = tk.Label(predictFrame, text="PREDICT", font=('Calibri', 14))
predictLabel.pack(fill='both', side='top', pady=20)

judulLabel = tk.Label(predictFrame, text="JUDUL BERITA", font=('Calibri', 14))
judulLabel.pack(fill='both', pady=5)

judulEntry = tk.Text(predictFrame, width=50, height=2, font=('Calibri', 12))
judulEntry.pack()
yscrollbar = tk.Scrollbar(predictFrame, orient='vertical', command=judulEntry.yview)
yscrollbar.pack(side='right', fill='y')
judulEntry["yscrollcommand"] = yscrollbar.set

narasiLabel = tk.Label(predictFrame, text="NARASI BERITA", font=('Calibri', 14))
narasiLabel.pack(fill='both', pady=5)

narasiEntry = tk.Text(predictFrame, width=50, height=10, font=('Calibri', 12))
narasiEntry.pack()
yscrollbar = tk.Scrollbar(predictFrame, orient='vertical', command=narasiEntry.yview)
yscrollbar.pack(side='right', fill='y')
narasiEntry["yscrollcommand"] = yscrollbar.set

buttonProcess = tk.Button(predictFrame, text="CHECK", command=helloCallBack, font=('Calibri', 14))
buttonProcess.pack(pady=10)

buttonClear = tk.Button(predictFrame, text="CLEAR", command=clearText, font=('Calibri', 14))
buttonClear.pack(pady=2)

entryPre = tk.StringVar()

preEntry = tk.Entry(predictFrame, state='disabled', textvariable=entryPre, justify='center',
                    font=('Calibri', 14))
preEntry.pack(ipadx=20, ipady=10, pady=20)

root.mainloop()
