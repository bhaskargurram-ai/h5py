Bug fixes
---------

* Compound datasets with a member of a custom floating-point type that h5py
  reads as a larger NumPy type, such as a 2-byte bfloat16 member read as
  ``float32``, can now be read and written. Previously opening such a dataset
  raised ``ValueError: NumPy dtype descriptor requires 4 bytes, cannot
  override to smaller itemsize of 2`` (:issue:`1825`).
