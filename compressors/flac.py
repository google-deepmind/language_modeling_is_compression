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

"""Implements a lossless compressor with FLAC."""

import audioop
import io

import pydub


def compress(data: bytes) -> bytes:
  """Returns data compressed with the FLAC codec.

  Args:
    data: Assumes 1 byte per sample (`sample_width`), meaning 256 possible
      values, and 1 channel and a `frame_rate` of 16kHz.
  """
  sample = pydub.AudioSegment(
      data=data,
      channels=1,
      sample_width=1,
      frame_rate=16000,
  )
  return sample.export(
      format='flac',
      parameters=['-compression_level', '12'],
  ).read()


def decompress(data: bytes) -> bytes:
  """Decompresses `data` losslessly using the FLAC codec.

  Args:
    data: The data to be decompressed. Assumes 2 bytes per sample (16 bit).

  Returns:
    The decompressed data. Assumes 1 byte per sample (8 bit).
  """
  sample = pydub.AudioSegment.from_file(io.BytesIO(data), format='flac')
  # FLAC assumes that data is 16 bit. However, since our original data is 8 bit,
  # we need to convert the samples from 16 bit to 8 bit (i.e., changing from two
  # channels to one channel with `lin2lin`) and add 128 since 16 bit is signed
  # (i.e., adding 128 using `bias`).
  return audioop.bias(audioop.lin2lin(sample.raw_data, 2, 1), 1, 128)
