# encryption.py
from tenseal import ts
import numpy as np

def encrypt_data(data):
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    encrypted_data = [ts.ckks_vector(context, x.flatten()) for x in data]
    return encrypted_data, context

def decrypt_data(encrypted_data, context):
    decrypted_data = [vec.decrypt().reshape(96, 96, 1) for vec in encrypted_data]
    return np.array(decrypted_data)
