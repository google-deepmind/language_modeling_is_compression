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

"""Implements an Arithmetic Encoder and Decoder."""

from typing import Any, Callable

import chex
import numpy as np

InputFn = Callable[[], int]
OutputFn = Callable[[int], None]
IOFn = InputFn | OutputFn


def _log_power_of_b(n: int, base: int) -> int:
  """Returns k assuming n = base ** k.

  We manually implement this function to be faster than a np.log or math.log,
  which doesn't assume n is an integer.

  Args:
    n: The integer of which we want the log.
    base: The base of the log.
  """
  log_n = 0
  while n > 1:
    n //= base
    log_n += 1
  return log_n


def _raise_post_terminate_exception(*args: Any, **kwargs: Any) -> None:
  """Dummy function that raises an error to ensure AC termination."""
  del args, kwargs
  raise ValueError(
      "Arithmetic encoder was terminated. "
      "Create a new instance for encoding more data. "
      "Do NOT use an output function that writes to the same data sink "
      "used by the output function of this instance. "
      "This will corrupt the arithmetic code as decoding relies on detecting "
      "when the compressed data stream is exhausted."
  )


class _CoderBase:
  """Arithmetic coder (AC) base class."""

  def __init__(self, base: int, precision: int, io_fn: IOFn):
    """Does initialization shared by AC encoder and decoder.

    Args:
      base: The arithmetic coder will output digits in {0, 1, ..., base - 1}.
      precision: Precision for internal state; on the average this will waste
        code space worth at most 1/log(base) * base ** -(precision - 2) digits
        of output per coding step.
      io_fn: Function to write digits to compressed stream/read digits from
        compressed stream.
    """
    chex.assert_scalar_in(base, 2, np.inf)
    chex.assert_scalar_in(precision, 2, np.inf)

    self._base: int = base
    self._base_to_pm1: int = int(base ** (precision - 1))
    self._base_to_pm2: int = int(base ** (precision - 2))
    self._io_fn = io_fn

    # NOTE: We represent the AC interval [0, 1) as rational numbers:
    #    [0, 1)
    #  ~ [self._low / base ** precision, (self._high + 1) / base ** precision)
    #  = [self._low / base ** precision, self._high / base ** precision],
    # where the we represent the upper bound *INCLUSIVE*. This is a subtle
    # detail required to make the integer arithmetic work correctly given that
    # all involved integers have `precision` digits in base `base`.
    self._low: int = 0
    self._high: int = int(base**precision) - 1
    self._num_carry_digits: int = 0
    self._code: int = 0

  def __str__(self) -> str:
    """Returns string describing internal state."""
    if self._base > 16:
      raise ValueError("`__str__` with `base` exceeding 16 not implmeneted.")

    p = 1 + _log_power_of_b(self._base_to_pm1, base=self._base)

    def _to_str(x: int) -> str:
      """Returns representation of `n` in base `self._base`."""
      digits = [(x // self._base**i) % self._base for i in range(p)]
      return f"{digits[-1]:x}<C:{self._num_carry_digits:d}>" + "".join(
          f"{d:x}" for d in digits[-2::-1]
      )

    return (
        f"[{_to_str(self._low)}, {_to_str(self._high)})  {_to_str(self._code)}"
    )

  def _get_intervals(self, pdf: np.ndarray) -> np.ndarray:
    """Partition the current AC interval according to the distribution `pdf`."""
    if (pdf < 0).any():
      raise ValueError(
          "Some probabilities are negative. Please make sure that pdf[x] > 0."
      )
    # Compute CPDF s.t. cpdf[x] = sum_y<x Pr[y] for 0 <= x < alphabet_size and
    # add a sentinel s.t. cpdf[alphabet_size] = 1.0. This partitions [0, 1) into
    # non-overlapping intervals, the interval [cpdf[x], cpdf[x + 1]) represents
    # symbol x. Since AC relies on an integer representation we rescale this
    # into the current AC range `high - low + 1` and quantise, this yields the
    # quantised CPDF `qcpdf`.
    width = self._high - self._low + 1
    qcpdf = (np.insert(pdf, 0, 0).cumsum() * width).astype(int)
    if (qcpdf[1:] == qcpdf[:-1]).any():
      raise ValueError(
          "Some probabilities are 0 after quantisation. Please make sure that:"
          " pdf[x] >= max(base ** -(precision - 2), np.dtype(x).eps) for any"
          " symbol by either preprocessing `pdf` or by increasing `precision`."
      )
    if qcpdf[-1] > width:
      raise ValueError(
          "Cumulative sum of probabilities exceeds 1 after quantisation. "
          "Please make sure that sum(pdf) <= 1.0 - eps, for a small eps > 0."
      )
    return self._low + qcpdf

  def _remove_matching_digits(self, low_pre_split: int, encoding: bool) -> None:
    """Remove matching most significant digits from AC state [low, high).

    This is the *FIRST* normalization step after encoding a symbol into the AC
    state.

    When encoding we write the most significant matching digits of the
    integer representation of [low, high) to the output, widen the integer
    representation of [low, high) including a (potential) queue of carry digits;
    when decoding we drop the matching most significant digits of the integer
    representation of [low, high), widen this interval and keep the current
    slice of the arithmetic code word `self._code` in sync.

    Args:
      low_pre_split: Value of `self._low` before encoding a new symbol into the
        AC state when `encoding` is True; abitrary, otherwise.
      encoding: Are we encoding (i.e. normalise by writing data) or decoding
        (i.e. normalise by reading data)?
    """

    def _shift_left(x: int) -> int:
      """Shift `x` one digit left."""
      return (x % self._base_to_pm1) * self._base

    while self._low // self._base_to_pm1 == self._high // self._base_to_pm1:
      if encoding:
        low_msd = self._low // self._base_to_pm1
        self._io_fn(low_msd)
        # Note that carry digits will only be written in the first round of this
        # loop.
        carry_digit = (
            self._base - 1 + low_msd - low_pre_split // self._base_to_pm1
        ) % self._base
        assert carry_digit in {0, self._base - 1} or self._num_carry_digits == 0
        while self._num_carry_digits > 0:
          self._io_fn(carry_digit)
          self._num_carry_digits -= 1
      else:
        self._code = _shift_left(self._code) + self._io_fn()
      self._low = _shift_left(self._low)
      self._high = _shift_left(self._high) + self._base - 1

  def _remove_carry_digits(self, encoding: bool) -> None:
    """Remove and record 2nd most significant digits from AC state [low, high).

    This is the *SECOND* normalization step after encoding a symbol into the AC
    state [low, high).

    If the AC state takes the form
       low  =   x B-1 B-1 ... B-1 u ...
       high = x+1   0   0       0 v ...
                  ^__  prefix __^
    where x, u and v are base-B digits then low and high can get arbitrarily (
    well, by means of infinite precision arithmetics) without matching. Since we
    work with finite precision arithmetics, we must make sure that this doesn't
    occour and we guarantee sufficient of coding range (`high - low`). To end
    this we detect the above situation and cut off the highlighted prefix above
    to widen the integer representation of [low, high) and record the number of
    prefix digits removed. When decoding we must similarly process the current
    slice of the arithmetic code word `self._code` to keep it in sync.

    Args:
      encoding: Are we encoding (i.e. normalise by writing data) or decoding
        (i.e. normalise by reading data)?
    """

    def _shift_left_keeping_msd(x: int) -> int:
      """Shift `x` except MSD, which remains in place, one digit left."""
      return x - (x % self._base_to_pm1) + (x % self._base_to_pm2) * self._base

    while self._low // self._base_to_pm2 + 1 == self._high // self._base_to_pm2:
      if encoding:
        self._num_carry_digits += 1
      else:
        self._code = _shift_left_keeping_msd(self._code) + self._io_fn()
      self._low = _shift_left_keeping_msd(self._low)
      self._high = _shift_left_keeping_msd(self._high) + self._base - 1

  def _process(self, pdf: np.ndarray, symbol: int | None) -> int:
    """Perform an AC encoding or decoding step and modify AC state in-place.

    Args:
      pdf: Probability distribution over input alphabet.
      symbol: Letter to encode from {0, 1, ..., pdf.size - 1} when encoding or
        `None` when decoding.

    Returns:
      y: `symbol` from above when encoding or decoded letter from {0, 1, ...,
        pdf.size - 1}.
    """

    encoding = symbol is not None
    intervals = self._get_intervals(pdf)
    if not encoding:
      symbol = np.searchsorted(intervals, self._code, side="right") - 1
    assert 0 <= symbol < pdf.size
    low_pre_split = self._low
    self._low, self._high = intervals[[symbol, symbol + 1]]
    # Due to integer arithmetics the integer representation of [low, high) has
    # an inclusive upper bound, so decrease high.
    self._high -= 1
    assert 0 <= self._low <= self._high < self._base_to_pm1 * self._base

    # Normalize the AC state.
    self._remove_matching_digits(low_pre_split=low_pre_split, encoding=encoding)
    assert 0 <= self._low <= self._high < self._base_to_pm1 * self._base
    assert encoding or self._low <= self._code <= self._high
    assert self._low // self._base_to_pm1 != self._high // self._base_to_pm1

    self._remove_carry_digits(encoding=encoding)
    assert 0 <= self._low <= self._high < self._base_to_pm1 * self._base
    assert encoding or self._low <= self._code <= self._high
    assert self._high - self._low > self._base_to_pm2

    return symbol

  @classmethod
  def p_min(cls, base: int, precision: int) -> float:
    """Get minimum probability supported by AC config."""
    # The leading factor 2 is supposed to account for rounding errors and
    # wouldn't be necessary given infinite float precision.
    return 2.0 * base ** -(precision - 2)


class Encoder(_CoderBase):
  """Arithmetic encoder."""

  def __init__(self, base: int, precision: int, output_fn: OutputFn):
    """Constructs arithmetic encoder.

    Args:
      base: The arithmetic coder will output digits in {0, 1, ..., base - 1}.
      precision: Precision for internal state; on the average this will waste
        code space worth at most 1/log(base) * base ** -(precision - 2) digits
        of output per coding step.
      output_fn: Function that writes a digit from {0, 1, ..., base - 1} to the
        compressed output.
    """
    super().__init__(base, precision, output_fn)

  def encode(self, pdf: np.ndarray, symbol: int) -> None:
    """Encodes symbol `symbol` assuming coding distribution `pdf`."""
    self._process(pdf, symbol)

  def terminate(self) -> None:
    """Finalizes arithmetic code."""
    # Write outstanding part of the arithmetic code plus one digit to uniquely
    # determine a code within the interval of the final symbol coded.
    self._io_fn(self._low // self._base_to_pm1)
    for _ in range(self._num_carry_digits):
      self._io_fn(self._base - 1)
    self.encode = _raise_post_terminate_exception
    self.terminate = _raise_post_terminate_exception


class Decoder(_CoderBase):
  """Arithmetic decoder."""

  def __init__(self, base: int, precision: int, input_fn: InputFn):
    """Constructs arithmetic decoder.

    Args:
      base: The arithmetic coder will output digits in {0, 1, ..., base - 1}.
      precision: Precision for internal state; on the average this will waste
        code space worth at most 1/log(base) * base ** -(precision - 2) digits
        of output per coding step.
      input_fn: Function that reads a digit from {0, 1, ..., base - 1} from the
        compressed input or returns `None` when the input is exhausted.
    """
    # Add padding to ensure the AC state is well-defined when decoding the last
    # symbol. Note that what exactly we do here depends on how encoder
    # termination is implemented (see `Encoder.terminate`).
    trailing_digits = (base - 1 for _ in range(precision - 1))

    def _padded_input_fn() -> int:
      """Reads digit from input padding the arithmetic code."""
      digit = input_fn()
      if digit is None:
        digit = next(trailing_digits)
      chex.assert_scalar_in(int(digit), 0, base - 1)
      return digit

    super().__init__(base, precision, _padded_input_fn)
    for _ in range(precision):
      self._code = self._code * base + _padded_input_fn()

  def decode(self, pdf: np.ndarray) -> int:
    return self._process(pdf, None)
