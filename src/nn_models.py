from typing import Union, Tuple, List, Optional
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers.schedules import InverseTimeDecay
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin


class NeuralNetwork(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        batch_size: int = 32,
        learning_rate_init: float = 0.1,
        learning_rate_decay_rate: float = 0.1,
        alpha: float = 0.0001,
        epochs: int = 100,
        class_weight: str = None,
        random_state: int = None,
    ):
        self._random_state = random_state
        self._seed_everything(random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.alpha = alpha
        self.epochs = epochs
        self.class_weight = class_weight

    @property
    def random_state(self) -> Union[int, None]:
        return self._random_state

    @random_state.setter
    def random_state(self, value: Union[int, None]):
        self._random_state = value
        self._seed_everything(value)

    def _seed_everything(self, value: Union[int, None]):
        if value is not None:
            np.random.seed(self.random_state)


    def set_model(self, X: Union[np.ndarray, pd.DataFrame]) -> Sequential:
        model = Sequential()
        model.add(
            Dense(
                self.hidden_layer_sizes[0],
                input_dim=X.shape[1],
                activation="relu",
                kernel_regularizer=l2(self.alpha),
            )
        )
        for layer_size in self.hidden_layer_sizes[1:]:
            model.add(
                Dense(
                    layer_size,
                    activation="relu",
                    kernel_regularizer=l2(self.alpha),
                )
            )
        model.add(Dense(1, activation="sigmoid"))
        lr_schedule = InverseTimeDecay(
            self.learning_rate_init,
            decay_steps=self.epochs,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False,
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=lr_schedule),
            metrics=["AUC"],
        )
        return model

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "NeuralNetwork":
        if self.class_weight == "balanced":
            self.class_weight = {0: 1 / sum(y == 0), 1: 1 / sum(y == 1)}
        self.model = self.set_model(X)
        self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            class_weight=self.class_weight if sample_weight is None else None,
            sample_weight=sample_weight,
            verbose=0,
        )
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        # prob = self.model.predict(X, verbose=0)
        prob = self.model(X.values, training=False)
        return np.concatenate([1 - prob, prob], axis=1)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.model(X.values, training=False) > 0.5

    def score(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> float:
        return self.model.evaluate(X, y, verbose=0)[1]