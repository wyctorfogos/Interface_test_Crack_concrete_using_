import mlflow.tensorflow
import tensorflow as tf

# Carregue o modelo .h5 em um objeto de modelo Keras (ou TensorFlow)
model = tf.keras.models.load_model("./models/modelo_treinado_224x224_ONLY_TRANSFER_LEARNING_MobileNetv2_softmax_4_categorias.h5")

# Inicie uma execução MLflow
with mlflow.start_run():
    # Registre o modelo no MLflow
    mlflow.tensorflow.log_model(model, artifact_path="model")

