# Authors: CommPy contributors
# License: BSD 3-Clause

""" Algorithms for Convolutional Codes """

from __future__ import division

import numpy as np

from commpy.utilities import dec2bitarray, bitarray2dec

__all__ = ['conv_encode']


def conv_encode(message_bits, trellis, termination = 'term', puncture_matrix=None):
    """
    Encode bits using a convolutional code.
    Parameters
    ----------
    message_bits : 1D ndarray containing {0, 1}
        Stream of bits to be convolutionally encoded.
    trellis: pre-initialized Trellis structure.
    termination: {'cont', 'term'}, optional
        Create ('term') or not ('cont') termination bits.
    puncture_matrix: 2D ndarray containing {0, 1}, optional
        Matrix used for the puncturing algorithm
    Returns
    -------
    coded_bits : 1D ndarray containing {0, 1}
        Encoded bit stream.
    """

    k = trellis.k
    n = trellis.n
    total_memory = trellis.total_memory
    rate = float(k)/n
    
    code_type = trellis.code_type

    if puncture_matrix is None:
        puncture_matrix = np.ones((trellis.k, trellis.n))

    number_message_bits = np.size(message_bits)
    
    if termination == 'cont':
        inbits = message_bits
        number_inbits = number_message_bits
        number_outbits = int(number_inbits/rate)
    else:
        # Initialize an array to contain the message bits plus the truncation zeros
        if code_type == 'rsc':
            inbits = message_bits
            number_inbits = number_message_bits
            number_outbits = int((number_inbits + k * total_memory)/rate)
        else:
            number_inbits = number_message_bits + total_memory + total_memory % k
            inbits = np.zeros(number_inbits, 'int')
            # Pad the input bits with M zeros (L-th terminated truncation)
            inbits[0:number_message_bits] = message_bits
            number_outbits = int(number_inbits/rate)

    outbits = np.zeros(number_outbits, 'int')
    if puncture_matrix is not None:
        # 修正前
        # p_outbits = np.zeros(number_outbits, 'int')
        # 修正後
        # 参考URL: https://github.com/veeresht/CommPy/pull/122
        number_punctured_bits = int(number_outbits * puncture_matrix.sum() / puncture_matrix.size)
        p_outbits = np.zeros(number_punctured_bits, 'int')
    else:
        p_outbits = np.zeros(int(number_outbits*
            puncture_matrix[0:].sum()/np.size(puncture_matrix, 1)), 'int')
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table

    # Encoding process - Each iteration of the loop represents one clock cycle
    current_state = 0
    j = 0

    for i in range(int(number_inbits/k)): # Loop through all input bits
        current_input = bitarray2dec(inbits[i*k:(i+1)*k])
        current_output = output_table[current_state][current_input]
        outbits[j*n:(j+1)*n] = dec2bitarray(current_output, n)
        current_state = next_state_table[current_state][current_input]
        j += 1

    if code_type == 'rsc' and termination == 'term':
        term_bits = dec2bitarray(current_state, trellis.total_memory)
        term_bits = term_bits[::-1]
        for i in range(trellis.total_memory):
            current_input = bitarray2dec(term_bits[i*k:(i+1)*k])
            current_output = output_table[current_state][current_input]
            outbits[j*n:(j+1)*n] = dec2bitarray(current_output, n)
            current_state = next_state_table[current_state][current_input]
            j += 1

    j = 0
    for i in range(number_outbits):
        if puncture_matrix[0][i % np.size(puncture_matrix, 1)] == 1:
            p_outbits[j] = outbits[i]
            j = j + 1

    return p_outbits
