import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as ci
import numpy as np
import commpy.utilities as util

from .coding.crc import calc_checksum
from .coding.turbo_code import turbo_encode, turbo_decode, map_decode


def encode_turbo(msg_bits, N):
    # トレリスの設定
    memory = np.array([4])

    # 生成多項式 1, 1+D, 1+D+D^2
    generator_matrix = np.array([[1, 3, 7]])
    code_type = 'rsc'
    feedback = np.array((3), ndmin=2)
    trellis1 = cc.Trellis(memory=memory,
            g_matrix=generator_matrix,
            code_type=code_type,
            feedback=feedback)
    trellis2 = cc.Trellis(memory=memory,
            g_matrix=generator_matrix,
            code_type=code_type,
            feedback=feedback)

    # インターリーバの設定
    L = N
    interleaver = ci.RandInterlv(length=L, seed=0)

    # encode
    [sys_stream, non_sys_stream1, non_sys_stream2] = turbo_encode(msg_bits, trellis1, trellis2, interleaver)
    assert np.all(msg_bits==sys_stream)
    sys_all = np.concatenate([sys_stream, non_sys_stream1, non_sys_stream2])
    #print(f'msg_bits\t(len={len(sys_stream):03d}): {sys_stream}')
    #print(f'e1 + e2\t(len={len(non_sys_stream1):03d}): {non_sys_stream1}')
    #print(f'e_bar1 + e_bar2\t(len={len(non_sys_stream2):03d}): {non_sys_stream2}')
    #print(f'sys_all\t(len={len(sys_all):03d}): {sys_all}')
    #print()
    return sys_stream, non_sys_stream1, non_sys_stream2, trellis1, interleaver


def check_recover(m, cs):
    eq = cs==calc_checksum(m)
    return np.all(eq)


def recover(m, e, e_bar, trellis_rate3, trellis_rate5, interleaver):
    MESSAGE_LENGTH = len(m)

    # 誤り訂正される前の m
    m_org = np.array(m)

    e_list = []
    e_bar_list = []

    # RSC code を併用して m を復元する
    start_idx = 0
    for idx in range(2):
        trellis = trellis_rate3 if idx == 0 else trellis_rate5
        #print('')
        #print(f'========== Recover by RSC decode (i = {idx+1})')
        #print('')
        e_ = e[start_idx:start_idx+MESSAGE_LENGTH]
        #print(f'e[{idx+1}]     (len={len(e_):03d}): {e_}')
        e_list.append(e_)
        # RSC decoding
        # TODO : map_decode でよいのか？
        _, m = map_decode(sys_symbols=m_org,
                non_sys_symbols=np.concatenate(e_list),
                trellis=trellis,
                noise_variance=1.0,
                # L_int が None の時の処理を真似て 0 で初期化する
                L_int=np.zeros(len(m_org)),
                mode='decode')
        #print(f'm\t(len={len(m):03d}): {m}')

        # Turbo code を併用して m を復元する
        #print('')
        #print(f'========== Recover by Turbo decode (i = {idx+1})')
        #print('')
        e_bar_ = e_bar[start_idx:start_idx+MESSAGE_LENGTH]
        #print(f'e_bar[{idx+1}] (len={len(e_bar_):03d}): {e_bar_}')
        e_bar_list.append(e_bar_)
        # Turbo decoding
        m = turbo_decode(sys_symbols=m_org,
                # e を渡す
                non_sys_symbols_1=np.concatenate(e_list),
                # e_bar を渡す
                non_sys_symbols_2=np.concatenate(e_bar_list),
                # 2 つ作っているがどちらも同じなので 1 つだけ渡す
                trellis=trellis,
                noise_variance=1.0,
                number_iterations=10,
                interleaver=interleaver,
                L_int=None)
        #print(f'm\t(len={len(m):03d}): {m}')

        start_idx += MESSAGE_LENGTH

    #print('')
    #print(f'm (len={len(m_org):03d}): {m_org}')
    return m
