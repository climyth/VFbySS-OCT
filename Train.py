from keras.callbacks import ModelCheckpoint, TensorBoard
from DataLoad import *
from Model import *

# Setup ===========================================
base_model_name = "InceptionV3"   # InceptionV3, InceptionV4, InceptionResnet
base_folder = "Z:/PaperResearch/VFbySS-OCT"
vf_file = "/train_data.xlsm"
weight_save_folder = "/Weights/" + base_model_name
pretrained_weights = ""   # 만약 없으면 "" 으로.
tensorboard_log_folder = "/Logs"
# ==================================================

# Data loading ===============================
print("Data loading...")
x_train, y_train, pids = LoadData(base_folder, base_folder + vf_file, False, "Train", "Train")

# model build ================================
model = GetModel(base_model_name)
if pretrained_weights != "":
    model.load_weights(base_folder + weight_save_folder + pretrained_weights)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# checkpoint
filepath = base_folder + weight_save_folder + "/" + base_model_name + "_24-2-improvement-{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir=tensorboard_log_folder, histogram_freq=1, write_graph=True, write_images=True)
callbacks_list = [checkpoint, tensorboard]

# Train ===========================================
model.fit(x_train, y_train, batch_size=32, validation_split=0.1, epochs=10000, callbacks=callbacks_list)
