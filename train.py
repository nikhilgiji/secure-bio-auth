# train.py
import setup
from data_preprocessing import load_data, preprocess_data, split_data
from model import create_model
from encryption import encrypt_data, decrypt_data
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
dataset_path = 'path/to/your/socofing/dataset'  # Update this path
data, labels = load_data(dataset_path)
data, labels, label_encoder = preprocess_data(data, labels)
X_train, X_test, y_train, y_test = split_data(data, labels)

# Encrypt data
X_train_encrypted, context = encrypt_data(X_train)
X_test_encrypted, _ = encrypt_data(X_test)

# Decrypt data for model training (homomorphic encryption training is advanced)
X_train_decrypted = decrypt_data(X_train_encrypted, context)
X_test_decrypted = decrypt_data(X_test_encrypted, context)

# Create model
input_shape = (96, 96, 1)
num_classes = len(label_encoder.classes_)
model = create_model(input_shape, num_classes)

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_decrypted, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test_decrypted, y_test)
print(f'Test accuracy: {test_acc}')

# Save model
model.save('fingerprint_recognition_model.h5')
