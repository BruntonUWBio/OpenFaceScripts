import pandas as pd


def get_merged_value(a, b, c, d):
    out_vals = []

    for a, b, c, d in zip(a, b, c, d):
        if pd.isnull(a):
            out_vals.append(b)
        elif pd.isnull(b):
            out_vals.append(a)
        elif c >= d:
            out_vals.append(a)
        elif d >= c:
            out_vals.append(b)

    return pd.Series(out_vals)


df1 = pd.DataFrame({'A': [1, 2, 1, 2], 'B': [1, 1, 1, 1], 'C': [1, 1, 2, 2]})
df2 = pd.DataFrame({'A': [1, 2], 'B': [2, 2], 'C': [2, 2]})
all_cols = list(df1.columns)
merged_cols = ['A', 'C']
other_cols = [x for x in all_cols if x not in merged_cols]
df3 = df1.merge(df2, 'outer', ['A', 'C'], suffixes=('_old', '_new'))
# print(df3)
assign_dict = {
    col: get_merged_value(df3[col + '_old'], df3[col + '_new'], df3.B_old,
                          df3.B_new)
    for col in other_cols
}
# print(assign_dict)

for col in other_cols:
    df3[col] = assign_dict[col]
# df3 = df3.assign(kwargs=assign_dict)
df3 = df3.drop(
    [x for x in list(df3.columns) if '_old' in x or '_new' in x], axis=1)
print(df3)
