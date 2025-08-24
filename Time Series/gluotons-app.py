"""Basit GluonTS DeepAR örneği.

Bu betik, GluonTS belgelerindeki "Simple Example" akışını
takip ederek AirPassengers verisinde bir DeepAR modeli eğitir,
tahmin üretir ve sonuçları çizer.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator


def load_air_passengers_dataframe() -> pd.DataFrame:
    """AirPassengers verisini pandas DataFrame olarak yükler."""
    csv_url = (
        "https://raw.githubusercontent.com/AileenNielsen/"
        "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    )
    dataframe = pd.read_csv(csv_url, index_col=0, parse_dates=True)
    return dataframe


def build_dataset(dataframe: pd.DataFrame) -> PandasDataset:
    """Veriyi GluonTS PandasDataset formatına dönüştürür."""
    dataset = PandasDataset(dataframe, target="#Passengers")
    return dataset


def train_deepar(training_data):
    """DeepAR modeli kurar ve eğitir; tahminci döndürür."""
    estimator = DeepAREstimator(
        prediction_length=12,
        freq="M",
        trainer_kwargs={"max_epochs": 5},
    )
    predictor = estimator.train(training_data)
    return predictor


def plot_forecasts(dataframe: pd.DataFrame, forecasts) -> None:
    """Gerçek değerleri ve çoklu pencere tahminlerini çizer."""
    plt.figure(figsize=(10, 5))
    plt.plot(dataframe["1954":], color="black")
    for forecast in forecasts:
        forecast.plot()
    plt.legend(["Gerçek değerler"], loc="upper left", fontsize="large")
    plt.title("DeepAR - AirPassengers Tahminleri")
    plt.tight_layout()
    plt.show()


def main() -> None:
    dataframe = load_air_passengers_dataframe()
    dataset = build_dataset(dataframe)

    training_data, test_generator = split(dataset, offset=-36)
    test_data = test_generator.generate_instances(
        prediction_length=12,
        windows=3,
    )

    predictor = train_deepar(training_data)
    forecasts = list(predictor.predict(test_data.input))

    plot_forecasts(dataframe, forecasts)


if __name__ == "__main__":
    main()


