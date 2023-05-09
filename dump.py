'''
            # Build the LSTM model
            model = Sequential()
            model.add(Embedding(vocab_size, 128, input_length=max_length-1))
            model.add(LSTM(256, return_sequences=True))
            model.add(LSTM(256))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(vocab_size, activation='softmax'))

            # Compile and summarize the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary()
        '''
        '''
        def build_model(hp):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length))
            
            for i in range(hp.Int('num_layers', 1, 3)):
                model.add(tf.keras.layers.LSTM(hp.Int(f'lstm_units_{i}', 128, 512, 32), return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False))
            
            model.add(tf.keras.layers.Dense(hp.Int('dense_units', 128, 512, 32), activation='relu'))
            model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), metrics=['accuracy'])
            
            return model

        tuner = RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=3,
            directory='my_dir',
            project_name='helloworld')

        tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

        top_10_models = tuner.get_best_models(num_models=10)
        '''
        
        self.model = Sequential()
        self.model.add(tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length))
            
        for i in range(hp.Int('num_layers', 1, 3)):
            self.model.add(LSTM(hp.Int(f'lstm_units_{i}', 128, 512, 32), return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False))
        
        model.add(tf.keras.layers.Dense(hp.Int('dense_units', 128, 512, 32), activation='relu'))
        model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), metrics=['accuracy'])
        