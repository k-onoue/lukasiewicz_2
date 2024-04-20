# import pandas as pd
# from mlxtend.frequent_patterns import apriori, association_rules


# def get_rules(df, min_support=0.1, min_threshold=0, conclusion_name='Outcome') -> pd.DataFrame:
#     freq_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=df.shape[1])
#     rules = association_rules(freq_itemsets, metric='confidence', min_threshold=min_threshold)
#     target_rules = rules[rules['consequents'].apply(lambda x: len(x) == 1 and conclusion_name in x)]
#     # target_rules['antecedents'] = target_rules['antecedents'].apply(lambda x: str(x).strip('frozenset({})'))
#     # target_rules['consequents'] = target_rules['consequents'].apply(lambda x: str(x).strip('frozenset({})'))
#     return target_rules


# class ArrangeRules:
#     def __init__(self, rules_df, feature_names=None, conclusion_name=None):
#         self.rules_df = rules_df
#         self.feature_names = feature_names

#         if not conclusion_name:
#             self.conclusion_name = 'Outcome'
#         else:
#             self.conclusion_name = conclusion_name

#         self.rules_extracted = None
#         self.rules_additional = None
#         self.KB = None

#     def extract_rules_from_df(self):
#         self.rules_extracted = []

#         for h in range(self.rules_df.shape[0]):
#             rule_info = self.rules_df.iloc[h]
#             antecedent = " ⊗ ".join(rule_info['antecedents'])
#             consequent = "".join(rule_info['consequents'])

#             if rule_info['lift'] - 1 > 0:
#                 rule = " → ".join([antecedent, consequent])
#             else:
#                 rule = " → ¬ ".join([antecedent, consequent])

#             self.rules_extracted.append(rule.split(" "))
        
#         return self.rules_extracted




#     def generate_rules_from_df(self):
#         if self.feature_names:
#             tmp_dict = {}
#             for item in self.feature_names:
#                 key, value = item.rsplit('_', 1)
#                 if key not in tmp_dict:
#                     tmp_dict[key] = []

#                 tmp_dict[key].append(item)
            
#             self.rules_additional = list(tmp_dict.values())
#             self.rules_additional = [' ⊕ '.join(rule) for rule in self.rules_additional]
#             self.rules_additional = [rule.split(' ') for rule in self.rules_additional]
#             return self.rules_additional
#         else:
#             return []


#     def construct_KB(self):
#         rules_extracted = self.extract_rules_from_df()

#         rules_additional = self.generate_rules_from_df()

#         self.KB = rules_extracted + rules_additional
#         return self.KB
    
#     def save_KB_as_txt(self, file_name):
#         if self.KB:
#             rules = [' '.join(rule) for rule in self.KB]


#         with open(file_name, 'w') as file:
#             for item in rules:
#                 file.write("%s\n" % item)


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def get_rules(df, min_support=0.1, min_threshold=0.5, conclusion_name='Outcome') -> pd.DataFrame:
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=df.shape[1])
    rules = association_rules(freq_itemsets, metric='confidence', min_threshold=min_threshold)
    # target_rules['antecedents'] = target_rules['antecedents'].apply(lambda x: str(x).strip('frozenset({})'))
    # target_rules['consequents'] = target_rules['consequents'].apply(lambda x: str(x).strip('frozenset({})'))
    return rules


class ArrangeRules:
    def __init__(self, rules_df, feature_names=None, conclusion_name=None):
        self.rules_df = rules_df
        self.feature_names = feature_names

        if not conclusion_name:
            self.conclusion_name = 'Outcome'
        else:
            self.conclusion_name = conclusion_name

        self.rules_extracted = None
        self.rules_additional = None
        self.KB = None

    def extract_rules_from_df(self):
        self.rules_extracted = []

        for h in range(self.rules_df.shape[0]):
            rule_info = self.rules_df.iloc[h]
            antecedent = " ⊗ ".join(rule_info['antecedents'])

            for consequent in self.rules_df.iloc[h]['consequents']:

                if rule_info['lift'] - 1 > 0:
                    rule = " → ".join([antecedent, consequent])
                else:
                    rule = " → ¬ ".join([antecedent, consequent])

                self.rules_extracted.append(rule.split(" "))
        
        return self.rules_extracted


    def generate_rules_from_df(self):
        if self.feature_names:
            tmp_dict = {}
            for item in self.feature_names:
                key, value = item.rsplit('_', 1)
                if key not in tmp_dict:
                    tmp_dict[key] = []

                tmp_dict[key].append(item)
            
            self.rules_additional = list(tmp_dict.values())
            self.rules_additional = [' ⊕ '.join(rule) for rule in self.rules_additional]
            self.rules_additional = [rule.split(' ') for rule in self.rules_additional]
            return self.rules_additional
        else:
            return []


    def construct_KB(self):
        rules_extracted = self.extract_rules_from_df()

        rules_additional = self.generate_rules_from_df()

        self.KB = rules_extracted + rules_additional
        return self.KB
    
    def save_KB_as_txt(self, file_name):
        if self.KB:
            rules = [' '.join(rule) for rule in self.KB]


        with open(file_name, 'w') as file:
            for item in rules:
                file.write("%s\n" % item)