# Total training time: 15999.80398440361 seconds
# Inference time for 10 samples: 1.1173293590545654 seconds
# loss: 1.7437 - category_level_1_loss: 0.2286 - category_level_2_loss: 0.2020 - category_level_1_sparse_categorical_accuracy: 0.9597 - category_level_2_sparse_categorical_accuracy: 0.9678 - val_loss: 21.3281 - val_category_level_1_loss: 1.4507 - val_category_level_2_loss: 2.6503 - val_category_level_1_sparse_categorical_accuracy: 0.6387 - val_category_level_2_sparse_categorical_accuracy: 0.4675 - lr: 1.0000e-04

from DataHandler import DataHandler
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, Embedding, BatchNormalization, Dropout, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import time
import pickle

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data.csv')
VOCUBULARY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocabulary.xlsx')
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_weights.keras')
PLOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plot.png')
HISTORY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'history.pkl')
RANDOM_STATE = 10
TRAIN_PERCENT = 0.8
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_ON_PLATEAU_PATIENCE = 2
BATCH_SIZE = 128
AUGMENT_TRAIN_DATASET = True

data_handler = DataHandler(vocabulary_path=VOCUBULARY_PATH)
(x_train, y_train), (x_test, y_test) = data_handler.load_data(dataset_path=DATA_PATH, train_percent=TRAIN_PERCENT, augment_train_dataset=AUGMENT_TRAIN_DATASET, random_state=RANDOM_STATE)

def build_model():
        input_title = Input(shape=(50,), name='title')
        input_content = Input(shape=(1500,), name='content')
        input_source = Input(shape=(5,), name='source')
        input_author = Input(shape=(5,), name='author')
        input_published = Input(shape=(6,), name='published')

        embedding_title = Embedding(input_dim=data_handler.get_vocabulary_size(), output_dim=100, input_length=50)(input_title)
        embedding_content = Embedding(input_dim=data_handler.get_vocabulary_size(), output_dim=100, input_length=1500)(input_content)
        
        embedding_title = Dropout(0.25)(embedding_title)
        embedding_content = Dropout(0.25)(embedding_content)

        LSTM_content = Bidirectional(LSTM(40, activation='tanh', recurrent_dropout=0.25, dropout=0.25))(embedding_content)
        LSTM_title = Bidirectional(LSTM(40, activation='tanh', recurrent_dropout=0.25, dropout=0.25))(embedding_title)
        
        dense_article = Dense(20, activation='relu')(concatenate([input_source, input_author, input_published, LSTM_title]))
        dense_article = BatchNormalization()(dense_article)
        dense_article = Dropout(0.25)(dense_article)
        
        output_category_level_1 = Dense(data_handler.get_category_1_vocabulary_size(), activation='softmax', name='category_level_1')(concatenate([dense_article, LSTM_content, LSTM_title]))
        output_category_level_2 = Dense(data_handler.get_category_2_vocabulary_size(), activation='softmax', name='category_level_2')(concatenate([dense_article, LSTM_content, LSTM_title]))

        model = Model(inputs=[input_source, input_title, input_content, input_author, input_published], outputs=[output_category_level_1, output_category_level_2])
        
        model.compile(optimizer='adam',
                loss={'category_level_1': 'sparse_categorical_crossentropy', 'category_level_2': 'sparse_categorical_crossentropy'},
                metrics={'category_level_1': 'sparse_categorical_accuracy', 'category_level_2': 'sparse_categorical_accuracy'},
                loss_weights={'category_level_1': 1.0, 'category_level_2': 7.5})

        return model

model = build_model()
model.summary()

# Callbacks
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=REDUCE_LR_ON_PLATEAU_PATIENCE, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True)

# Train the model and calculate the training time
start_time = time.time()
history = model.fit(
        x_train, 
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data= (x_test, y_test),
        callbacks=[reduce_lr_on_plateau, early_stopping]
)
end_time = time.time()
total_training_time = end_time - start_time
print("Total training time: {} seconds".format(total_training_time))

model.save_weights(MODEL_WEIGHTS_PATH)

# Calculate inference time
test_samples = {key: value[:10] for key, value in x_test.items()}
start_time = time.time()
predictions = model.predict(test_samples)
end_time = time.time()
inference_time = end_time - start_time
print("Inference time for 10 samples: {} seconds".format(inference_time))

with open(HISTORY_PATH, 'wb') as file:
    pickle.dump(history.history, file)
