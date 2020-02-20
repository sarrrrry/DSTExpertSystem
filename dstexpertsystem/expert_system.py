
from abc import ABCMeta, abstractmethod
class BaseQuestions(metaclass=ABCMeta):
    @property
    @abstractmethod
    def KnowledgeBase(self):
        pass

    def __init__(self):
        pass

from dstexpertsystem.read_csv import read_csv
# 質問表と尤度
class Q_Animal(BaseQuestions):
    class KnowledgeBase:
        def __init__(self, path):
            self.df = read_csv(path)

        @property
        def answers(self):
            return list(self.df.index)

        def calc_likelihood_from_(self, answer, weights, sort=False):
            match_level = self.df.loc[answer].copy()
            def calc_basic_likelihood(match_level):
                def func(weight):
                    return match_level[str(weight)]
                return func
            func = calc_basic_likelihood(match_level)
            likelihood = weights.apply(func)

            if sort:
                # sort
                likelihood = likelihood.sort_values(ascending=True)  # 降順で並び替え

            # 正規化
            likelihood /= likelihood.max()

            return likelihood


    def __init__(self, match_table_path, convert_table_path):
        df = read_csv(match_table_path)
        self.df = df

        self.KB = self.KnowledgeBase(convert_table_path)

    @property
    def expect_answers(self):
        return self.KB.df.index

    def __getitem__(self, idx):
        self.latest_idx = idx
        #TODO: idxの例外処理
        return self.df.index[idx]

    def __len__(self):
        return len(self.df.index)

    def be_answered(self, answer):
        if self.latest_idx < 0:
            print("CALL the function 'question()'")
            return None

        # 基本尤度の割当
        likelihood = self.KB.calc_likelihood_from_(
            answer=answer,
            weights=self.df.iloc[self.latest_idx],
            sort=True
        )

        # import pandas as pd
        # lh = pd.Series([0.01, 0.90, 1.00], index=['コアラ', 'タヌキ', 'ライオン'])
        return likelihood


import numpy as np
def calc_mass_from_likelihood(likelihood):
    # 尤度を降順に並び替え
    lh = likelihood.copy()
    lh_sorted = lh.sort_values(ascending=True)

    # 基本確率の割当
    masses = []
    masses = [[list(lh.index), lh.iloc[0]]]

    for idx, _ in enumerate(lh.index[:-1]):
        mass_val = 0.0
        mass_val = lh[lh.index[idx+1]] - lh[lh.index[idx]]
        mass_val = np.round(mass_val, 4)  # 小数点第4位まで
        mass_key = list(lh.index[idx+1:])
        masses.append([mass_key, mass_val])

    return masses

import pandas as pd
class DST:
    # Dempster & Shafer theorem
    def __init__(self):
        self.prior_masses = []

    def update(self, posterior_masses):
        if self.prior_masses:
            self.__update(posterior_masses)
        else:
            self.prior_masses = posterior_masses

    def __update(self, posterior_masses):

        # 算出に必要なDFを作成する
        ## 焦点要素と基本確立に切り分ける
        def expand_to_df(prior, post, idx):
            # idx番目のリストだけを取得する
            prior_row = [p[idx] for p in prior]
            post_row = [p[idx] for p in post]

            # テーブル形式に拡張する
            prior = [prior_row for _ in range(len(post_row))]
            post = [post_row for _ in range(len(prior_row))]

            prior_df = pd.DataFrame(prior).T
            post_df = pd.DataFrame(post)
            return prior_df, post_df

        prior_elem_df, post_elem_df = expand_to_df(
            prior=self.prior_masses, 
            post=posterior_masses,
            idx=0
        )
        prior_mass_df, post_mass_df = expand_to_df(
            prior=self.prior_masses, 
            post=posterior_masses,
            idx=1
        )

        # 要素のintersectionを算出
        ## 内部要素をappend出来るように1次元拡張する
        prior_elem_df = prior_elem_df.applymap(lambda x: [x])
        post_elem_df = post_elem_df.applymap(lambda x: [x])
        ## 各要素の0番目と1番目の積集合を算出
        inter_elem = prior_elem_df + post_elem_df
        inter_elem = inter_elem.applymap(lambda x: set(x[0]).intersection(set(x[1])))

        # 確率のDFを作成
        prod_mass = prior_mass_df * post_mass_df

        # -------------------------
        # DFを合成して事後確率を計算
        # -------------------------
        # 空集合の事後確率を計算
        focal_element = inter_elem.astype('bool')  # 焦点要素: 空集合ではない各要素
        is_emptyset = ~focal_element
        emptyset_probability = prod_mass[is_emptyset].sum().sum()
        normalization_value = 1 - emptyset_probability

        # 各要素の事後確率を計算
        post_prob = []
        unique = inter_elem.applymap(lambda x: list(x))
        unique = np.unique(unique.values.ravel())
        for key in unique:
            if not key:
                continue
            mask = np.full(inter_elem.shape, set(key))
            df = prod_mass[inter_elem == mask]
            probability = df.sum().sum()
            probability /= normalization_value
            post_prob.append([list(key), probability])
        # 浮動小数点誤差があるので手動調整
        sum_prob = sum([p[1] for p in post_prob])
        post_prob = [[p[0], p[1]/sum_prob] for p in post_prob]

        # 事前分布を更新
        del self.prior_masses
        self.prior_masses = post_prob

    @property
    def prob(self):
        prob = self.prior_masses
        prob = [[p[0], p[1]] for p in prob if len(p[0])==1]
        prob.sort(key=lambda x: x[1], reverse=True)
        sum_prob = sum(p[1] for p in prob)
        prob.append(['判断不能', 1-sum_prob])
        return prob

class ExpertSystem:
    def __init__(self, questions, prob_rule):
        print("Expert System is initialized")
        self.idx = -1
        self.Qs = questions
        self.prob_rule = prob_rule

        self.expect_answers = [[str(idx+1), str(val)] for idx, val in enumerate(self.Qs.expect_answers)]

    def __iter__(self):
        print(" called iter ")
        return self

    def __next__(self):
        """
        iterations
            from -> 'self.idx + 1'
            to   -> 'len(self.Qs)'
        e.g.)
            self.idx     => -1
            len(self.Qs) => 3
            loop         => [0, 1, 2]
        """

        if (self.idx >= len(self.Qs)-1):
            raise StopIteration
        self.idx += 1

        return self

    def start(self):
        print("*** Expert System is being started ***")

    def question(self):
        print()

        # ユーザへ質問をする
        question = str(self.Qs[self.idx])
        msg = 'ANSWER => ' + ''.join([f"| {idx}: {ans} " for idx, ans in self.expect_answers]) + '|'
        print(f"Q{self.idx+1} : {question}")
        print(msg)
        print(f">>> ", end="")

    def be_answered(self, answer):
        exp = dict(self.expect_answers)
        if answer in exp.values():
            pass
        elif answer in exp.keys():
            answer = exp[answer]
        else:
            msg = ''.join([f"| {idx}: {ans} " for idx, ans in self.expect_answers]) + '|'
            print(f"WARNING: ANSWER MUST BE IN THE ABOVE")
            print(">>> ", end="")
            self.be_answered(input())

        lh = questions.be_answered(answer)

        # 基本確立の計算をする
        masses = calc_mass_from_likelihood(lh)

        # Dempster & Shaferの更新
        self.prob_rule.update(masses)

    def log_prob(self):
        print(self.prob_rule.prob)


if __name__ == "__main__":
    mt_path = "./data/match_table.csv"
    ct_path = "./data/convert_table.csv"
    questions = Q_Animal(
        match_table_path=mt_path,
        convert_table_path=ct_path
    )
    prob_rule = DST()
    es = ExpertSystem(questions, prob_rule)
    es.start()

    for Q in es:
        Q.question()
        answer = input()
        # print()
        # answer = "はい"
        Q.be_answered(answer)

        Q.log_prob()

    print('This is the end of Expert System ...')

