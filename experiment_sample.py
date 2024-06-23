



df_origin_1 = pd.read_csv(file_path_1, index_col=0).reset_index(drop=True)
X_origin_1 = df_origin_1.drop(["Outcome"], axis=1)
y_origin_1 = df_origin_1["Outcome"]

df_origin_2 = pd.read_csv(file_path_2, index_col=0).reset_index(drop=True)
X_origin_2 = df_origin_2.drop(["Outcome"], axis=1)
y_origin_2 = df_origin_2["Outcome"]
print(df_origin_1.head())
print(df_origin_2.head())

# 実験設定
settings = {
    'path': os.path.join(project_dir_path, "experiment_result/tmp/version_1"),
    'source_paths': [file_path_1, file_path_2],
    'experiment_name': 'pima_indian_diabetes_test',
    'seed': 42,
    'n_unsupervised': 15,
    'c1': 10,
    'c2': 10,
    'result': {}
}

if not os.path.exists(settings['path']):
    os.makedirs(settings['path'])
    os.makedirs(os.path.join(settings['path'], "rules"))

# データの分割
train_idx, test_idx = train_test_split(df_origin_1.index, test_size=0.2, random_state=settings['seed'])

print("\n\n\n\n\n")
print(f"fold: 1 of 1")

settings['result'][f'fold_0'] = {}

# ルールの獲得 (RuleFit Classifier (continuous))
X_train = X_origin_2.copy().iloc[train_idx].values
y_train = y_origin_2.copy().iloc[train_idx].values
X_test = X_origin_2.copy().iloc[test_idx].values
y_test = y_origin_2.copy().iloc[test_idx].values

feature_names = list(X_origin_2.columns)

rulefit = RuleFitClassifier(
    rfmode='classify',
    tree_generator=RandomForestClassifier(random_state=42),
    random_state=42,
    exp_rand_tree_size=False
)

rulefit.fit(X_train, y_train, feature_names=feature_names)

# ルールの整形
rules_df = rulefit.get_rules(exclude_zero_coef=True)
rule_processor = ArrangeRules(
    rules_df,
    feature_names=feature_names,
    conclusion_name="Outcome"
)
KB_origin = rule_processor.construct_KB()
rule_processor.save_KB_as_txt(os.path.join(settings['path'], f'rules/rules_0.txt'))

rules_tmp = []
for rule in KB_origin:
    if "Outcome" in rule:
        tmp = {}
        for idx, item in enumerate(rule):
            if not is_symbol(item):
                if idx == 0 or rule[idx - 1] != '¬':
                    tmp[item] = 1
                elif item != "Outcome":
                    tmp[item] = 0
                else:
                    tmp[item] = -1

        rules_tmp.append(tmp)

rule_violation_check = {}

for h, rule in enumerate(rules_tmp):
    outcome = rule['Outcome']

    condition_parts = [
        f"{column} == {value}"
        for column, value in rule.items()
        if column != "Outcome"
    ]
    condition = " & ".join(condition_parts)

    satisfying_idxs = df_origin_2.loc[test_idx].query(condition).index

    rule_violation_check[h] = (satisfying_idxs, outcome)

# テストデータ
df_tmp = df_origin_1.copy().iloc[test_idx, :]
df_tmp = df_tmp.rename(columns={'Outcome': 'target'})
df_tmp['target'] = df_tmp['target'].replace(0, -1)

input_for_test = {
    'data': df_tmp,
    'rule': rule_violation_check
}

# 訓練データ
L = {}
for col_name in df_origin_2.columns:
    df_new = X_origin_1.copy().iloc[train_idx, :]
    df_new['target'] = df_origin_2[col_name].replace(0, -1)
    L[col_name] = df_new

np.random.seed(seed=settings['seed'])
arr_u = np.random.rand(settings['n_unsupervised'], X_origin_1.shape[1])
U = {key: arr_u for key in L.keys()}

S = {key: np.vstack([df.drop(['target'], axis=1).values, arr_u]) for key, df in L.items()}

# ルール
KB_origin = KB_origin

# パラメータ
len_j = len(L)
len_l = len(train_idx)
len_u = settings['n_unsupervised']
len_s = len_l + len_u

len_h = len(KB_origin)
len_i = len_u * 2

# モデルの学習 6（論理制約付きモデル）
input_luka_1 = {
    'L': L,
    'U': U,
    'S': S,
    'len_j': len_j,
    'len_l': len_l,
    'len_u': len_u,
    'len_s': len_s,
    'len_h': len_h,
    'len_i': len_i,
    'c1': settings['c1'],
    'c2': settings['c2'],
    'KB_origin': KB_origin,
    'target_predicate': 'Outcome',
    'kernel_function': "logistic regression",
}

problem_instance = SetupPrimal(input_luka_1)
objective_function, constraints = problem_instance.main()
problem = cp.Problem(objective_function, constraints)
result = problem.solve(verbose=True)

problem_info = problem_instance.problem_info
p_name = problem_instance.problem_info['target_predicate']
p_trained = problem_instance.problem_info['predicates_dict'][p_name]
y_pred = p_trained(X_test).value

y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)

result = evaluate_model(
    y_test,
    y_pred,
    y_pred_interpreted,
    input_for_test,
    test_idx
)

settings['result'][f'fold_0']['logistic regression (L)'] = result

# 実験結果の保存
with open(os.path.join(settings['path'], 'result.json'), 'w') as f:
    json.dump(settings, f, indent=4)
