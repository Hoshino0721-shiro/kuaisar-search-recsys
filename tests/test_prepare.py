import unittest
from src.prepare_data import milliseconds, split_of, TRAIN_END, VALID_END


class PreparationTests(unittest.TestCase):
    def test_integer_milliseconds(self):
        self.assertEqual(milliseconds('1684738202782.0'),1684738202782)
        for value in ('NaN','Infinity','123.5'):
            with self.assertRaises(ValueError):
                milliseconds(value)

    def test_half_open_splits(self):
        self.assertEqual(split_of(TRAIN_END-1),'train')
        self.assertEqual(split_of(TRAIN_END),'valid')
        self.assertEqual(split_of(VALID_END-1),'valid')
        self.assertEqual(split_of(VALID_END),'test')
