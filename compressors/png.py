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

"""Implements a lossless compressor with PNG."""

import io
import math

from PIL import Image


def _get_the_two_closest_factors(n: int) -> tuple[int, int]:
  """Returns the 2 closest factors (square root if `n` is a perfect square)."""
  a = round(math.sqrt(n))
  while n % a > 0:
    a -= 1
  return a, n // a


def compress(data: bytes) -> bytes:
  """Compresses `data` losslessly using the PNG format.

  The data, which is a sequence of bytes, is reshaped into a
  as-close-to-square-as-possible image before compression with 8-bit pixels
  (grayscale).

  Args:
    data: The data to be compressed.

  Returns:
    The compressed data.
  """
  # Compute the height and width of the image.
  size = _get_the_two_closest_factors(len(data))

  # Load the image using 8-bit grayscale pixels.
  image = Image.frombytes(mode='L', size=size, data=data)

  with io.BytesIO() as buffer:
    image.save(
        buffer,
        format='PNG',
        optimize=True,
    )
    return buffer.getvalue()


def decompress(data: bytes) -> bytes:
  """Decompresses `data` losslessly using the PNG format.

  To apply the PNG format, the `data` is treated as the compressed sequence of
  bytes from an image consisting of 8-bit pixels (grayscale).

  Args:
    data: The data to be decompressed.

  Returns:
    The decompressed data.
  """
  return Image.open(io.BytesIO(data)).tobytes()
