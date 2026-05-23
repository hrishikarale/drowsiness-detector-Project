import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# ── Config
DATA_DIR   = r"data\CEW"
IMG_SIZE   = (24, 24)
BATCH_SIZE = 32
EPOCHS     = 15
MODEL_PATH = r"models\eye_state_classifier.h5"
os.makedirs("models", exist_ok=True)

# ── Data loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False
)

# ── Normalise
normalise = layers.Rescaling(1.0 / 255)
train_ds  = train_ds.map(lambda x, y: (normalise(x), y))
val_ds    = val_ds.map(lambda x, y: (normalise(x), y))
train_ds  = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds    = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# ── Model
model = keras.Sequential([
    layers.Input(shape=(24, 24, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(1,  activation='sigmoid'),
])

model.summary()

# ── Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks
early_stop = keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True,
    monitor='val_accuracy'
)
checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor='val_accuracy'
)

# ── Train
print("\nStarting training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stop, checkpoint]
)

# ── Evaluate
print("\nEvaluating on validation set...")
loss, acc = model.evaluate(val_ds)
print(f"Val accuracy: {acc*100:.2f}%")
print(f"Val loss:     {loss:.4f}")

# ── Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(r"models\training_history.png")
plt.show()
print("Training history saved to models/training_history.png")

# ── Verify
print("\nVerifying saved model...")
loaded = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")
loaded.summary()