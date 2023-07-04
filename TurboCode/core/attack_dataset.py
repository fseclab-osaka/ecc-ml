import numpy as np


class AttackDataset():
    def __init__(self, input_shape, n_data_points, n_categories, data_points_seed=42):
        """
        Parameters
        ----------
        input_shape : tuple[int]
            入力の形状。
        n_data_points : int
            攻撃データセットのデータ数。
        n_categories : int
            攻撃データセットのカテゴリ数。
        data_points_seed : int
            攻撃データセットの data point 生成シード。
        """
        self.input_shape = input_shape
        self.n_data_points = n_data_points
        self.n_categories = n_categories
        self.data_points_seed = data_points_seed
        self.__labels = None
        return

    @property
    def sub_block_length(self):
        """
        1 つのラベルに対応するサブブロック（ビット列）の長さ。
        埋め込むメッセージはこの長さでサブブロックに分割される。
        """
        return int(np.floor(np.log2(self.n_categories)))

    @property
    def max_category_idx(self):
        """
        攻撃データセットに含まれるカテゴリインデックスの最大値。
        実際のデータセットに含むカテゴリでも、長さ sub_block_length のビット列で表せないものがある。
        """
        return int('1' * self.sub_block_length, 2)

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, labels):
        labels_array = np.array(labels)
        if labels_array.ndim != 1:
            raise ValueError('labels は 1 次元配列にしてください。')

        if labels_array.shape[0] != self.n_data_points:
            raise ValueError(f'labels は data point と同じ要素数 {self.n_data_points} としてください。')

        self.__labels = labels_array

    def get_data_points(self):
        rng = np.random.default_rng(seed=self.data_points_seed)
        return rng.integers(0, 255, [self.n_data_points, *self.input_shape], dtype=np.uint8)

    def sub_block_to_category(self, sub_block):
        """
        メッセージのサブブロック（ビット配列）に対応するカテゴリを求める。

        Parameters
        ----------
        sub_block : numpy.ndarray
            サブブロック（ビット列）。各要素は 0 または 1 となる。

        Returns
        -------
        category_idx : int
            サブブロック（ビット列）に対応するカテゴリのインデックス。
        """
        # 2 進文字列を作成
        #   ex) [0, 1, 1] -> '011'
        sub_block_str = ''.join(map(str, sub_block.astype(np.uint8).tolist()))
        # 2 進文字列を数値に変換
        category_idx = int(sub_block_str, 2)
        return category_idx

    def category_to_sub_block(self, category_idx):
        """
        カテゴリのインデックスに対応するサブブロック（ビット列）を求める。

        Parameters
        ----------
        category_idx : int
            カテゴリのインデックス。

        Returns
        -------
        sub_block : numpy.ndarray
            カテゴリのインデックスに対応するサブブロック（ビット列）。各要素は 0 または 1 となる。
            長さはプロパティ sub_block_length となり、左側は 0 埋めされる。
        """
        # sub_block_length 桁になるよう 0 埋めした 2 進文字列
        sub_block_str = f'{category_idx:0{self.sub_block_length}b}'
        # numpy.ndarray に変換
        #   ex) '011' -> [0, 1, 1]
        sub_block = np.array([int(e) for e in sub_block_str], dtype=np.uint8)
        return sub_block
