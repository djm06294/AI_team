# %%
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

SIZE = 128
root_folder_train = 'C:\\Users\\user\\Desktop\\roots\\train'
root_folder_test = 'C:\\Users\\user\\Desktop\\roots\\test'
# image = Image.open(os.path.join(root_folder_train, 'root1_220818\\root1_220818174850.jpg'))
# numpydata = np.asarray(image)
# print(numpydata.shape)
# print(type(numpydata))
# plt.imshow(numpydata)


def find_time(file_name, st_file_name) :
    st_date = datetime.strptime('20' + st_file_name[6:18], '%Y%m%d%H%M%S')
    date = datetime.strptime('20' + file_name[6:18], '%Y%m%d%H%M%S')

    hours = (date - st_date).total_seconds() / 3600.0
    # print(hours)

    if(hours<12) : return 0
    elif(hours<24) : return 1
    elif(hours<36) : return 2
    elif(hours<48) : return 3
    elif(hours<60) : return 4
    elif(hours<72) : return 5
    else : return 6


def find_middle_crop_resize(image, size) :
    img_cropped = image.crop((490,0,1490,1000))
    numpydata = np.asarray(img_cropped)

    # 이미지 윗 부분 crop
    # for i in range(0, 200, 10) :
    #     line = numpydata[i,:]
    #     if np.where(line > np.mean(line))[0].shape[0] < 230 : break #숫자가 높을수록 위쪽을 덜 자름
    
    # 이미지 윗 부분 crop 안하기
    line = numpydata[0,:]
    i=0    

    # x = np.array(range(line.shape[0]))
    # y = line
    # plt.plot(x,y)

    p = (np.max(line) + np.mean(line))/2

    idx_middle = 490 + int(np.mean(np.where(line > p)))
    img_cropped = image.crop((idx_middle-500,i,idx_middle+500,1000+i))

    img_resize = img_cropped.resize((size, size))
    return np.asarray(img_resize)


def load_images(root_folder, image_arrays, image_times) :
    sum = 0
    # root_folder의 폴더 하나씩 확인하기
    for folder_name in os.listdir(root_folder) :
        folder_path = os.path.join(root_folder, folder_name)
        print(folder_name)
        i = 0

        # directory만 처리
        if not os.path.isdir(folder_path) : continue

        for file_name in os.listdir(folder_path) :
            file_path = os.path.join(folder_path, file_name)

            # readme.txt 있는 폴더의 이미지는 사용 안함
            # if any(glob.glob(folder_path + '\\*.txt')) :    
            #     print("txt file found") 
            #     break

            # 첫 이미지명
            if i==0 : st_file_name = file_name

            # .jpg만 처리
            if not file_name.endswith('.jpg') : continue

            try :   
                image = Image.open(file_path).convert('L')
            except :
                print('error reading file ', file_path)
                continue
            
            # 모든 이미지 저장하기 (크기는 128 x 128)
            np_image = find_middle_crop_resize(image, SIZE)
            image_arrays.append(np_image)
            # 시간차 계산해서 저장하기
            image_times.append(find_time(file_name, st_file_name))
            
            i = i+1

        sum = sum + i
    return sum      # 총 저장된 사진 개수



#%%
image_arrays_train = []
image_times_train = []
sum_train = 0
image_arrays_test = []
image_times_test = []
sum_test = 0

sum_train = load_images(root_folder_train, image_arrays_train, image_times_train)
sum_test = load_images(root_folder_test, image_arrays_test, image_times_test)

#
import tensorflow as tf
from tensorflow import keras

x_train = np.array(image_arrays_train)
y_train = np.array(image_times_train)
x_test = np.array(image_arrays_test)
y_test = np.array(image_times_test)

print(x_train.shape)
#
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

n = 1
plt.imshow(x_train[0], cmap='Greys', interpolation='nearest')
plt.show()


x_train = x_train.reshape(x_train.shape[0], SIZE,SIZE,1)
x_test = x_test.reshape(x_test.shape[0], SIZE,SIZE,1)
print(x_train.shape)
input_shape = (SIZE,SIZE,1)


print(y_train[:10])
num_classes = 7
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[:10])


#
import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
np.random.seed(7)


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.summary()

#
batch_size = 128
epochs = 4
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1, 
                 validation_data=(x_test, y_test))

# 
plt.figure(figsize=(12,8))
#plt.plot(hist.history['loss’])
#plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
#plt.legend(['loss','val_loss', 'acc','val_acc’])
plt.legend(['acc','val_acc'])
plt.show()

# %% 
import random
predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)
test_labels = np.argmax(y_test, axis=1)
wrong_result = []
for n in range(0, len(test_labels)) :
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)
samples = random.choices(population = wrong_result, k=16)
count = 0
nrows = ncols = 4
plt.figure(figsize=(12,8))
for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(x_test[n].reshape(SIZE, SIZE), cmap='Greys', interpolation = 'nearest')
    tmp = "Label: " + str(test_labels[n]) + ", Prediction: " + str(predicted_labels[n])
    plt.title(tmp)
plt.tight_layout()
plt.show()