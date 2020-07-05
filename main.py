from __future__ import absolute_import, division, print_function, unicode_literals
from fastai.vision import *
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense


class Ai:
    ERROR_SKIP = 'skip'

    @staticmethod
    def main():
        model = Ai()
        keras.datasets.reuters
        # updater = Updater(local_params['telegram_key'], use_context=True)
        # dispatcher: Dispatcher = updater.dispatcher
        # dispatcher.add_error_handler(model.error_handler)
        # # main functionality
        # dispatcher.add_handler(CommandHandler("start", model.start), 3)
        #
        # # Start the Bot
        # updater.start_polling()
        # updater.idle()

    # HANDLERS

    @staticmethod
    def image_category_fastai():
        classes = ['goat', 'sheep']
        path = os.path.abspath('images')
        for c in classes:
            verify_images(path + '/' + c, delete=True, max_size=500)
        data = ImageDataBunch \
            .from_folder(path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4) \
            .normalize(imagenet_stats)

        learn = create_cnn(data, models.resnet34, metrics=error_rate)
        learn.fit_one_cycle(5)
        # learn.save('stage-1')
        # learn.export()
        # learner_obj: Learner = load_learner('./', 'export.pkl')
        img = open_image(path + '/goat/00000022.jpg')
        y, pred, raw_pred = learn.predict(img)
        print(raw_pred)

    @staticmethod
    def number_predict():
        c = np.array([-40, -10, 0, 8, 15, 22, 38])  # Celsius temperature
        f = np.array([-40, 14, 32, 46, 59, 72, 100])  # Fahrenheit temperature
        model = keras.Sequential()
        model.add(Dense(units=1, input_shape=(1,), activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))
        history = model.fit(c, f, epochs=500, verbose=0)
        plt.plot(history.history['loss'])
        plt.grid(True)
        plt.show()  # print trainig success chart
        print(model.predict([100]))  # predict a Celsius value for 100Fahrenheit

    @staticmethod
    def image_category():
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=10)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print("\nAccuracy: ", test_acc)
        predictions = model.predict(test_images)
        print("\nPredictions: ", predictions[0])


if __name__ == '__main__':
    Ai.main()
