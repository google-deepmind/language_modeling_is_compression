# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements data loaders."""

import audioop
from collections.abc import Iterator
import itertools
import os.path
import random
import urllib.request
import zipfile

import numpy as np
import tensorflow_datasets as tfds

from language_modeling_is_compression import constants


def _get_librispeech_dataset():
  return tfds.load('librispeech', split='train_clean100')


def _get_imagenet_dataset():
  return tfds.load('imagenet2012')['full']


def get_enwik9_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
  """Returns an iterator for enwik9 data."""
  if not os.path.exists('enwik9'):
    # Downloading and extracting the dataset.
    urllib.request.urlretrieve(
        'https://mattmahoney.net/dc/enwik9.zip',
        'enwik9.zip',
    )
    with zipfile.ZipFile('enwik9.zip', 'r') as zip_ref:
      zip_ref.extract('enwik9')

  all_chunks = []
  with open('enwik9', 'rb') as file:
    for _ in range(num_chunks):
      all_chunks.append(file.read(sequence_length))
  return iter(all_chunks)


def _extract_audio_patches(sample: bytes) -> Iterator[bytes]:
  patches = np.array_split(
      np.frombuffer(sample, dtype=np.uint8),
      range(
          constants.CHUNK_SIZE_BYTES,
          len(sample),
          constants.CHUNK_SIZE_BYTES,
      ),
  )
  if len(patches[-1]) != constants.CHUNK_SIZE_BYTES:
    patches.pop()
  return map(lambda x: x.tobytes(), patches)


def get_librispeech_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns an iterator for librispeech data."""
  # Convert samples from 16 bit to 8 bit (i.e., changing from two channels to
  # one channel with `lin2lin`), adding 128 since 16 bit is signed (i.e., adding
  # 128 using `bias`).
  librispeech_dataset = map(
      lambda x: audioop.bias(audioop.lin2lin(x['speech'], 2, 1), 1, 128),
      _get_librispeech_dataset().as_numpy_iterator(),
  )
  idx = 0
  for data in librispeech_dataset:
    for patch in _extract_audio_patches(data):
      if idx == num_chunks:
        return
      yield patch
      idx += 1


def get_random_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns an iterator for random data."""
  for _ in range(num_chunks):
    yield random.randbytes(constants.CHUNK_SIZE_BYTES)


def _rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
  return np.mean(image, axis=-1).astype(np.uint8)


def _extract_image_patches(image: np.ndarray) -> Iterator[bytes]:
  h, w = constants.CHUNK_SHAPE_2D
  height, width = image.shape

  for row, col in itertools.product(range(height // h), range(width // w)):
    yield image[row * h : (row + 1) * h, col * w : (col + 1) * w].tobytes()


def get_imagenet_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns a iterator for imagenet data."""
  imagenet_dataset = map(
      lambda x: _rgb_to_grayscale(x['image']),
      _get_imagenet_dataset().as_numpy_iterator(),
  )
  idx = 0
  for data in imagenet_dataset:
    for patch in _extract_image_patches(data):
      if idx == num_chunks:
        return
      yield patch
      idx += 1


GET_DATA_GENERATOR_FN_DICT = {
    'enwik9': get_enwik9_iterator,
    'imagenet': get_imagenet_iterator,
    'librispeech': get_librispeech_iterator,
    'random': get_random_iterator,
}
