import binascii

import numpy as np


def calc_checksum(msg):
    """
    Parameters
    ----------
    msg : numpy.ndarray
        メッセージ（ビット列）。各要素は 0 または 1 となる。

    Returns
    -------
    checksum : numpy.ndarray
        メッセージに対するチェックサム（ビット列）。長さ 32 で、各要素は 0 または 1 となる。
    """
    # メッセージの 2 進文字列を作成
    #   ex) [0, 1, 1] -> '011'
    msg_str = ''.join(map(str, msg.astype(np.uint8).tolist()))
    # bytes へ変換
    msg_bytes = msg_str.encode('utf-8')
    # CRC32 チェックサムを求めて 2 進文字列化
    checksum_str = f'{binascii.crc32(msg_bytes):032b}'
    # numpy.ndarray に変換
    #   ex) '011' -> [0, 1, 1]
    checksum = np.array([int(e) for e in checksum_str], dtype=np.uint8)
    return checksum
