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

"""Utility functions."""

import chex
import numpy as np


def bits_to_bytes(bits: str) -> tuple[bytes, int]:
  """Returns the bytes representation of bitstream and number of padded bits."""
  # Pad the string with zeros if the length is not a multiple of 8.
  padded_bits = bits.zfill((len(bits) + 7) // 8 * 8)
  num_padded_bits = len(padded_bits) - len(bits)

  # Split the string into 8-bit chunks.
  chunks = [padded_bits[i : i + 8] for i in range(0, len(padded_bits), 8)]

  # Convert each chunk to an integer and then to a byte.
  bytes_data = bytes([int(chunk, base=2) for chunk in chunks])

  return bytes_data, num_padded_bits


def bytes_to_bits(data: bytes, num_padded_bits: int = 0) -> str:
  """Returns the bitstream of bytes data accounting for padded bits."""
  return ''.join([bin(byte)[2:].zfill(8) for byte in data])[num_padded_bits:]


def right_shift_bytes_by_one(data: bytes) -> tuple[bytes, int]:
  """Returns right-shifted bytes, i.e., divided by 2, and the number of bytes.

  Our language models were trained on ASCII data. However, not all bytes can be
  decoded to ASCII, so we set the most significant bit (MSB) to 0, to ensure
  that we can decode the data to ASCII.

  However, for certain data types (e.g., images), masking the MSB and leaving
  the rest of the byte unchanged will destroy the structure of the data. Thus,
  we instead divide the number by two (i.e., we shift the bits to the right by
  one).

  Args:
    data: The bytes to be shifted.
  """
  return bytes([byte >> 1 for byte in data]), len(data)


def zero_most_significant_bit_if_not_ascii_decodable(
    data: bytes,
) -> tuple[bytes, int]:
  """Returns ascii-decodable data & the number of zeroed most significant bits.

  Our language models were trained on ASCII data. However, not all bytes can be
  decoded to ASCII, so we set the most significant bit (MSB) to 0, to ensure
  that we can decode the data to ASCII.

  Args:
    data: The bytes to be shifted.
  """
  masked_bits = 0
  masked_data = list()

  for byte in data:
    if chr(byte).isascii():
      masked_data.append(byte)
    else:
      masked_bits += 1
      masked_data.append(byte & 0x7F)

  return bytes(masked_data), masked_bits


def normalize_pdf_for_arithmetic_coding(pdf: chex.Array) -> chex.Array:
  """Normalizes the probabilities for arithmetic coding.

  Arithmetic coding converts the floating-point pdf to integers to avoid
  numerical issues. To that end, all pdf values need to be larger than the
  machine epsilon (to yield different integer values) and the sum of the pdf
  cannot exceed 1 (minus some precision tolerance).

  Args:
    pdf: The probabilities to be normalized.

  Returns:
    The normalized probabilities.
  """
  machine_epsilon = np.finfo(np.float32).eps
  # Normalize the probabilities to avoid floating-point errors.
  pdf = pdf / np.cumsum(pdf)[-1]
  # Ensure all probabilities are sufficiently large to yield distinct cdfs.
  pdf = (1 - 2 * pdf.shape[0] * machine_epsilon) * pdf + machine_epsilon
  return pdf
