import pandas as pd
import os

path = "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/train/csv_files/" \
       "with_wham_noise_res_more_informative.csv" #todo- general path
df = pd.read_csv(path)
df = df[df.speaker2_id != df.speaker1_id]

same_gender_bool = False
only_male = False
only_female = False
same_gender_few_diff_bool = True

if same_gender_bool:
    print(df[df.same_gender == 1].index)
    df = df[df.same_gender == 1]
    print(os.path.splitext(path)[0] +  '_only_same_gender.csv')
    new_path = os.path.splitext(path)[0] +  '_only_same_gender.csv'
    ##save processing
    df.to_csv(new_path)
elif only_male:
    print(df[(df.speaker1_gender == 1) & (df.speaker0_gender == 1)].index)
    df = df[(df.speaker1_gender == 1) & (df.speaker0_gender == 1)]
    print(os.path.splitext(path)[0] +  '_only_male.csv')
    new_path = os.path.splitext(path)[0] +  '_only_male.csv'
    ##save processing
    df.to_csv(new_path)
elif only_female:
    print(df[(df.speaker1_gender == 0) & (df.speaker0_gender == 0)].index)
    df = df[(df.speaker1_gender == 0) & (df.speaker0_gender == 0)]
    print(os.path.splitext(path)[0] +  '_only_female.csv')
    new_path = os.path.splitext(path)[0] +  '_only_female.csv'
    ##save processing
    df.to_csv(new_path)
elif same_gender_few_diff_bool:
    df_same_gender = df[df.same_gender == 1]
    num_same_gender = df_same_gender.shape[0]
    df_diff_gender = df[df.same_gender == 0][:int(num_same_gender * 0.2)]
    df_mix = pd.concat([df_same_gender, df_diff_gender])
    print(os.path.splitext(path)[0] +  '_only_same_gender_with_20p_diff.csv')
    new_path = os.path.splitext(path)[0] +  '_only_same_gender_with_20p_diff.csv'
    ##save processing
    df_mix.to_csv(new_path)