from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

class MLP(Model):
    
    def __init__(self, output_units):
        super(MLP, self).__init__()
        
        # Remove input_shape from Dense layers
        self.dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(0.3)
        
        self.dense2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))
        self.batch_norm2 = BatchNormalization()
        self.dropout2 = Dropout(0.3)
        
        self.dense3 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))
        self.batch_norm3 = BatchNormalization()
        self.dropout3 = Dropout(0.3)
        
        self.dense4 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))
        self.batch_norm4 = BatchNormalization()
        self.dropout4 = Dropout(0.3)
        
        self.output_layer = Dense(output_units, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.batch_norm3(x, training=training)
        x = self.dropout3(x, training=training)
        
        x = self.dense4(x)
        x = self.batch_norm4(x, training=training)
        x = self.dropout4(x, training=training)
        
        return self.output_layer(x)

