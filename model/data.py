import pandas as pd


def downsample(df):
    # return df
    print("Before downlsampling", df.size)
    original = df[df.purchase == 1]
    sampled = df[df.purchase == 0].sample(n=len(original))
    concatenated = pd.concat([original, sampled])
    print("After downlsampling", concatenated.size)
    return concatenated
