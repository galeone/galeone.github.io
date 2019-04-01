import tensorflow_datasets.public_api as tfds
import tensorflow as tf


class Crypto15(tfds.core.GeneratorBasedBuilder):
    """Crypto15 dataset: snapshot captured every 15 minutes
    of the status of some of the most common crypto currencies.
    """

    VERSION = tfds.core.Version("0.1.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Crypto15 dataset: snapshot captured every 15 minutes "
                "of the status of some of the most common crypto currencies"
            ),
            features=tfds.features.FeaturesDict(
                {
                    "symbol": tfds.features.Text(),
                    "timestamp": tfds.features.Tensor(shape=(1,), dtype=tf.uint64),
                    "price_btc": tfds.features.Tensor(shape=(1,), dtype=tf.float64),
                    "price_usd": tfds.features.Tensor(shape=(1,), dtype=tf.float64),
                    "day_volume": tfds.features.Tensor(shape=(1,), dtype=tf.uint64),
                    "market_cap": tfds.features.Tensor(shape=(1,), dtype=tf.uint64),
                    "percent_change_1h": tfds.features.Tensor(
                        shape=(1,), dtype=tf.float64
                    ),
                    "percent_change_24h": tfds.features.Tensor(
                        shape=(1,), dtype=tf.float64
                    ),
                    "percent_change_7d": tfds.features.Tensor(
                        shape=(1,), dtype=tf.float64
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        # Downloads the data and defines the splits
        # dl_manager is a tfds.download.DownloadManager that can be used to
        # download and extract URLs
        pass  # TODO

    def _generate_examples(self):
        # Yields examples from the dataset
        pass  # TODO
