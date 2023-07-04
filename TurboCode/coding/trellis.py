import commpy.channelcoding.convcode as cc
import numpy as np


TURBO_CODE_RATES = ['1/3', '1/5']


def generate_trellis(turbo_code_rate):
    """
    Parameters
    ----------

    Returns
    -------
    """
    memory = np.array([4])
    # 符号化率により generator_matrix のサイズが変わる
    # 行数k、列数nとする時、rate=k/nである
    if turbo_code_rate == '1/5':
        rsc_code_rate = '1/3'
        # 生成多項式 1, 1+D, 1+D+D^2
        generator_matrix = np.array([[1, 3, 7]])
    elif turbo_code_rate == '1/3':
        rsc_code_rate = '1/2'
        # 生成多項式 1, 1+D
        generator_matrix = np.array([[1, 3]])
    else:
        raise ValueError(f'{TURBO_CODE_RATES}から選択してください。')
    feedback = np.array((3), ndmin=2)

    trellis = cc.Trellis(memory=memory,
                         g_matrix=generator_matrix,
                         code_type='rsc',
                         feedback=feedback)

    return trellis
