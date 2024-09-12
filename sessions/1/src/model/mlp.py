import tensorflow as tf

class MLP(tf.keras.Model):
    
    def __init__(self, output_units):
        super(MLP, self).__init__()
        
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        
        self.dense2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.dense3 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        
        self.dense4 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.dropout4 = tf.keras.layers.Dropout(0.3)
        
        self.output_layer = tf.keras.layers.Dense(output_units, activation='sigmoid')

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

