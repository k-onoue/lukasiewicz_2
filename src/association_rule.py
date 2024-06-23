import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def get_rules(df, min_support=0.1, min_threshold=0.5, conclusion_name='Outcome') -> pd.DataFrame:
    """
    Extracts association rules from a DataFrame using the Apriori algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing transaction data.
    min_support : float, optional
        The minimum support threshold for the Apriori algorithm (default is 0.1).
    min_threshold : float, optional
        The minimum threshold for the confidence metric (default is 0.5).
    conclusion_name : str, optional
        The name of the conclusion item to filter the rules (default is 'Outcome').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the filtered association rules.

    Notes
    -----
    - The function uses the Apriori algorithm to find frequent itemsets and then extracts association rules.
    - It filters the rules to include only those where the conclusion_name is either in the antecedents or consequents.
    """
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=df.shape[1])
    rules = association_rules(freq_itemsets, metric='confidence', min_threshold=min_threshold)

    # conclusion_name が前件または後件に含まれているという条件でフィルタリング
    rules = rules[rules['antecedents'].apply(lambda x: 'Outcome' in x) 
                  | rules['consequents'].apply(lambda x: 'Outcome' in x)]
    
    return rules

class ArrangeRules:
    """
    A class to arrange and process association rules.

    Attributes
    ----------
    rules_df : pd.DataFrame
        The DataFrame containing association rules.
    feature_names : list of str, optional
        A list of feature names.
    conclusion_name : str
        The name of the conclusion item (default is 'Outcome').
    rules_extracted : list of list of str
        The extracted rules.
    rules_additional : list of list of str
        Additional rules generated from feature names.
    KB : list of list of str
        The knowledge base containing all rules.

    Methods
    -------
    extract_rules_from_df() -> list:
        Extracts rules from the DataFrame.
    generate_rules_from_df() -> list:
        Generates additional rules from feature names.
    construct_KB() -> list:
        Constructs the knowledge base from extracted and additional rules.
    save_KB_as_txt(file_name) -> None:
        Saves the knowledge base to a text file.
    """
    def __init__(self, rules_df, feature_names=None, conclusion_name=None):
        """
        Initializes the ArrangeRules class with a DataFrame of rules and optional feature names.

        Parameters
        ----------
        rules_df : pd.DataFrame
            The DataFrame containing association rules.
        feature_names : list of str, optional
            A list of feature names (default is None).
        conclusion_name : str, optional
            The name of the conclusion item (default is 'Outcome').
        """
        self.rules_df = rules_df.copy()
        self.feature_names = feature_names

        if not conclusion_name:
            self.conclusion_name = 'Outcome'
        else:
            self.conclusion_name = conclusion_name

        self.rules_extracted = None
        self.rules_additional = None
        self.KB = None

    def extract_rules_from_df(self):
        """
        Extracts rules from the DataFrame.

        Returns
        -------
        list of list of str
            The extracted rules.

        Notes
        -----
        - This method processes the DataFrame to create a list of rules.
        - The rules are split into antecedents and consequents based on the lift value.
        """
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
        """
        Generates additional rules from feature names.

        Returns
        -------
        list of list of str
            The generated additional rules.

        Notes
        -----
        - This method creates additional rules based on feature names.
        - The features are grouped and combined using the '⊕' operator.
        """
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
        """
        Constructs the knowledge base from extracted and additional rules.

        Returns
        -------
        list of list of str
            The knowledge base containing all rules.

        Notes
        -----
        - This method combines the extracted rules and additional rules to create the knowledge base.
        """
        rules_extracted = self.extract_rules_from_df()
        self.KB = rules_extracted
        return self.KB
    
    def save_KB_as_txt(self, file_name):
        """
        Saves the knowledge base to a text file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the knowledge base.

        Notes
        -----
        - This method writes the rules in the knowledge base to a text file.
        """
        if self.KB:
            rules = [' '.join(rule) for rule in self.KB]

        with open(file_name, 'w') as file:
            for item in rules:
                file.write("%s\n" % item)
