from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import tensorflow

class PrepareBaseModel:
    def __init__(self, config : PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tensorflow.keras.applications.vgg16.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )

        tensorflow.keras.models.save_model(model = self.model, filepath = self.config.base_model_path)

    

    @staticmethod
    def _prepare_full_model_(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till>0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False


        flatten_in = tensorflow.keras.layers.Flatten()(model.output)
        prediction = tensorflow.keras.layers.Dense(
            units = classes,
            activation = 'softmax'
        )(flatten_in)

        full_model = tensorflow.keras.models.Model(
            inputs = model.input,
            outputs = prediction
        )

        full_model.compile(
            optimizer = tensorflow.keras.optimizers.SGD(learning_rate = learning_rate),
            loss = tensorflow.keras.losses.CategoricalCrossentropy(),
            metrics = ['accuracy']
        )

        full_model.summary()

        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model_(
            model = self.model, 
            classes = self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        tensorflow.keras.models.save_model(model = self.full_model, filepath = self.config.updated_base_model_path)
