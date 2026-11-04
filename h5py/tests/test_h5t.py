# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

import numpy as np

import h5py
from h5py import h5t

from .common import TestCase, ut


class TestCompound(ut.TestCase):

    """
        Feature: Compound types can be created from Python dtypes
    """

    def test_ref(self):
        """ Reference types are correctly stored in compound types (issue 144)
        """
        dt = np.dtype([('a', h5py.ref_dtype), ('b', '<f4')])
        tid = h5t.py_create(dt, logical=True)
        t1, t2 = tid.get_member_type(0), tid.get_member_type(1)
        self.assertEqual(t1, h5t.STD_REF_OBJ)
        self.assertEqual(t2, h5t.IEEE_F32LE)
        self.assertEqual(tid.get_member_offset(0), 0)
        self.assertEqual(tid.get_member_offset(1), h5t.STD_REF_OBJ.get_size())

    def test_out_of_order_offsets(self):
        size = 20
        type_dict = {
            'names': ['f1', 'f2', 'f3'],
            'formats': ['<f4', '<i4', '<f8'],
            'offsets': [0, 16, 8]
        }

        expected_dtype = np.dtype(type_dict)

        tid = h5t.create(h5t.COMPOUND, size)
        for name, offset, dt in zip(
                type_dict["names"], type_dict["offsets"], type_dict["formats"], strict=True
        ):
            tid.insert(
                name.encode("utf8") if isinstance(name, str) else name,
                offset,
                h5t.py_create(dt)
            )

        self.assertEqual(tid.dtype, expected_dtype)
        self.assertEqual(tid.dtype.itemsize, size)


def _bfloat16_type():
    """2-byte custom float (bfloat16): sign bit 15, 8 exponent bits at 7,
    7 mantissa bits at 0, bias 127."""
    tid = h5t.IEEE_F32LE.copy()
    tid.set_fields(15, 7, 8, 0, 7)
    tid.set_size(2)
    tid.set_ebias(127)
    tid.lock()
    return tid


# bfloat16 bit patterns of pi, 1/3 and 1, and their values as float32
BF16_BITS = np.array([0x4049, 0x3eab, 0x3f80], dtype='<u2')
BF16_VALUES = np.array([3.140625, 0.333984375, 1.0], dtype='<f4')


class TestCompoundCustomFloat(TestCase):

    """
        Feature: Compound members of a custom float type that is read as a
        larger NumPy type (issue 1825)
    """

    def make_file(self, name, comp, raw):
        """Write a chunked dataset of compound type comp from the raw
        bytes in the structured array raw, using the low-level API."""
        fname = self.mktemp().encode()
        fid = h5py.h5f.create(fname)
        space = h5py.h5s.create_simple((len(raw),))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((1,))
        dset = h5py.h5d.create(fid, name, comp, space, dcpl)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, raw, comp)
        del dset, space, dcpl, fid
        return fname.decode()

    def test_single_member(self):
        """ The issue's layout: one array member of bfloat16 """
        bf16 = _bfloat16_type()
        comp = h5t.create(h5t.COMPOUND, 2)
        comp.insert(b'values', 0, h5t.array_create(bf16, (1,)))
        raw = BF16_BITS.view([('values', '<u2', (1,))])
        fname = self.make_file(b'x', comp, raw)

        with h5py.File(fname, 'r') as f:
            dset = f['x']
            self.assertEqual(dset.dtype, np.dtype([('values', '<f4', (1,))]))
            np.testing.assert_array_equal(dset[:]['values'][:, 0], BF16_VALUES)
            np.testing.assert_array_equal(dset['values'][:, 0], BF16_VALUES)
            np.testing.assert_array_equal(
                dset.fields('values')[:][:, 0], BF16_VALUES)

    def test_mixed_members(self):
        """ A widened member alongside members that keep their size """
        bf16 = _bfloat16_type()
        comp = h5t.create(h5t.COMPOUND, 8)
        comp.insert(b'tag', 0, h5t.STD_U16LE)
        comp.insert(b'values', 2, bf16)
        comp.insert(b'half', 4, h5t.IEEE_F16LE)
        comp.insert(b'count', 6, h5t.STD_I16LE)
        raw = np.zeros(3, dtype=[('tag', '<u2'), ('values', '<u2'),
                                 ('half', '<f2'), ('count', '<i2')])
        raw['tag'] = [10, 20, 30]
        raw['values'] = BF16_BITS
        raw['half'] = [0.5, -1.5, 2.0]
        raw['count'] = [-1, 0, 7]
        fname = self.make_file(b'x', comp, raw)

        with h5py.File(fname, 'r+') as f:
            dset = f['x']
            self.assertEqual(
                dset.dtype,
                np.dtype([('tag', '<u2'), ('values', '<f4'),
                          ('half', '<f2'), ('count', '<i2')]))
            data = dset[:]
            np.testing.assert_array_equal(data['tag'], raw['tag'])
            np.testing.assert_array_equal(data['values'], BF16_VALUES)
            np.testing.assert_array_equal(data['half'], raw['half'])
            np.testing.assert_array_equal(data['count'], raw['count'])
            np.testing.assert_array_equal(dset['values'], BF16_VALUES)
            np.testing.assert_array_equal(dset[1], data[1])

            # writing goes back through the same conversion
            data['values'] = [1.0, -2.0, 0.5]
            data['count'] = [1, 2, 3]
            dset[:] = data
            np.testing.assert_array_equal(dset['values'], [1.0, -2.0, 0.5])
            np.testing.assert_array_equal(dset['count'], [1, 2, 3])
            np.testing.assert_array_equal(dset['half'], raw['half'])

    def test_standard_members_keep_offsets(self):
        """ Compounds without a widened member keep the HDF5 layout """
        comp = h5t.create(h5t.COMPOUND, 12)
        comp.insert(b'half', 0, h5t.IEEE_F16LE)
        comp.insert(b'tag', 4, h5t.STD_U16LE)
        comp.insert(b'single', 8, h5t.IEEE_F32LE)
        expected = np.dtype({'names': ['half', 'tag', 'single'],
                             'formats': ['<f2', '<u2', '<f4'],
                             'offsets': [0, 4, 8], 'itemsize': 12})
        self.assertEqual(comp.dtype, expected)
        self.assertEqual(comp.dtype.itemsize, comp.get_size())


class TestTypeFloatID(TestCase):
    """Test TypeFloatID."""

    def test_custom_float_promotion(self):
        """Custom floats are correctly promoted to standard floats on read."""

        # This test uses the low-level API, so we need names as byte strings
        test_filename = self.mktemp().encode()
        dataset = b'DS1'
        dataset2 = b'DS2'
        dataset3 = b'DS3'
        dataset4 = b'DS4'
        dataset5 = b'DS5'

        dims = (4, 7)

        wdata = np.array([[-1.50066626e-09,   1.40062184e-09,   1.81216819e-10,
                           4.01087163e-10,   4.27917257e-10,  -7.04858394e-11,
                           5.74800652e-10],
                          [-1.50066626e-09,   4.86579665e-10,   3.42879503e-10,
                           5.12045517e-10,   5.10226528e-10,   2.24190444e-10,
                           3.93356459e-10],
                          [-1.50066626e-09,   5.24778443e-10,   8.19454726e-10,
                           1.28966349e-09,   1.68483894e-10,   5.71276360e-11,
                           -1.08684617e-10],
                          [-1.50066626e-09,  -1.08343556e-10,  -1.58934199e-10,
                           8.52196536e-10,   6.18456397e-10,   6.16637408e-10,
                           1.31694833e-09]], dtype=np.float32)

        wdata2 = np.array([[-1.50066626e-09,   5.63886715e-10,  -8.74251782e-11,
                            1.32558853e-10,   1.59161573e-10,   2.29420039e-10,
                            -7.24185156e-11],
                           [-1.50066626e-09,   1.87810656e-10,   7.74889486e-10,
                            3.95630195e-10,   9.42236511e-10,   8.38554115e-10,
                            -8.71978045e-11],
                           [-1.50066626e-09,   6.20275387e-10,   7.34871719e-10,
                            6.64840627e-10,   2.64662958e-10,   1.05319486e-09,
                            1.68256520e-10],
                           [-1.50066626e-09,   1.67347025e-10,   5.12045517e-10,
                            3.36513040e-10,   1.02545528e-10,   1.28784450e-09,
                            4.06089384e-10]], dtype=np.float32)

        # Create a new file using the default properties.
        fid = h5py.h5f.create(test_filename)
        # Create the dataspace.  No maximum size parameter needed.
        space = h5py.h5s.create_simple(dims)

        # create a custom type with larger bias
        mytype = h5t.IEEE_F16LE.copy()
        mytype.set_fields(14, 9, 5, 0, 9)
        mytype.set_size(2)
        mytype.set_ebias(53)
        mytype.lock()

        dset = h5py.h5d.create(fid, dataset, mytype, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata)

        del dset

        # create a custom type with larger exponent
        mytype2 = h5t.IEEE_F16LE.copy()
        mytype2.set_fields(15, 9, 6, 0, 9)
        mytype2.set_size(2)
        mytype2.set_ebias(53)
        mytype2.lock()

        dset = h5py.h5d.create(fid, dataset2, mytype2, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)

        del dset

        # create a custom type which reimplements 16-bit floats
        mytype3 = h5t.IEEE_F16LE.copy()
        mytype3.set_fields(15, 10, 5, 0, 10)
        mytype3.set_size(2)
        mytype3.set_ebias(15)
        mytype3.lock()

        dset = h5py.h5d.create(fid, dataset3, mytype3, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)

        del dset

        # create a custom type with larger bias
        mytype4 = h5t.IEEE_F16LE.copy()
        mytype4.set_fields(15, 10, 5, 0, 10)
        mytype4.set_size(2)
        mytype4.set_ebias(258)
        mytype4.lock()

        dset = h5py.h5d.create(fid, dataset4, mytype4, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)

        del dset

        # create a dataset with long doubles
        dset = h5py.h5d.create(fid, dataset5, h5t.NATIVE_LDOUBLE, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)

        # Explicitly close and release resources.
        del space
        del dset
        del fid

        f = h5py.File(test_filename, 'r')

        # ebias promotion to float32
        values = f[dataset][:]
        np.testing.assert_array_equal(values, wdata)
        self.assertEqual(values.dtype, np.dtype('<f4'))

        # esize promotion to float32
        values = f[dataset2][:]
        np.testing.assert_array_equal(values, wdata2)
        self.assertEqual(values.dtype, np.dtype('<f4'))

        # regular half floats
        dset = f[dataset3]
        try:
            self.assertEqual(dset.dtype, np.dtype('<f2'))
        except AttributeError:
            self.assertEqual(dset.dtype, np.dtype('<f4'))

        # ebias promotion to float64
        dset = f[dataset4]
        self.assertEqual(dset.dtype, np.dtype('<f8'))

        # long double floats
        dset = f[dataset5]
        self.assertEqual(dset.dtype, np.longdouble)
