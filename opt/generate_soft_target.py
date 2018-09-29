# config: utf8

import os
import warnings
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img


class CustomIDGenerator(ImageDataGenerator):

    def __init__(self, src_model):
        super().__init__(src_model)
        self.src_model = src_model

    def flow(self, x,
             y=None,
             batch_size=32, shuffle=True,
             sample_weight=None, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', subset=None):

        return CustomNumpyArrayIterator(
            x, y, self.src_model, self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset)


class CustomNumpyArrayIterator(NumpyArrayIterator):

    def __init__(self, x, y, src_model, image_data_generator,
                 batch_size=32, shuffle=False, sample_weight=None,
                 seed=None, data_format='channels_last',
                 save_to_dir=None, save_prefix='', save_format='png',
                 subset=None, dtype='float32'):
        super().__init__(src_model)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=self.dtype)
        for i, j in enumerate(index_array):
            x = self.x[j]
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(
                x.astype(self.dtype), params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        output = (batch_x if batch_x_miscs == []
                  else [batch_x] + batch_x_miscs,)
        if self.y is None:
            if batch_x_miscs == []:
                y = self.src_model.prediction(batch_x)
                output += (y)
            return output
        output += (self.y[index_array],)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output
