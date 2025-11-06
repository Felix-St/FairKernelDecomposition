import pandas as pd
import numpy as np

import folktables


def load_dataset(name):
    if name == "Crime":
        str_protected = "racepctblack"

        column_names = list(pd.read_csv("src/data/CC/names.csv").columns)

        data_df = pd.read_csv(
            "src/data/CC/communities.data", sep=",", names=column_names, na_values="?"
        )

        target = data_df["ViolentCrimesPerPop"]

        protected = data_df[str_protected]

        # Remove non-predictive, features with many missing values and target
        to_remove = (
            ["state", "county", "community", "communityname", "fold"]
            + list(data_df.columns[data_df.isnull().any()])
            + ["ViolentCrimesPerPop"]
            + [str_protected]
        )

        data_df.drop(columns=to_remove, inplace=True, axis="columns")

        X = data_df.to_numpy()
        y = target.to_numpy()
        p = protected.to_numpy()

        return X, y, p
    elif name == "CrimeMulti":
        str_protected = ["racepctblack","racePctWhite"]

        column_names = list(pd.read_csv("src/data/CC/names.csv").columns)

        data_df = pd.read_csv(
            "src/data/CC/communities.data", sep=",", names=column_names, na_values="?"
        )

        target = data_df["ViolentCrimesPerPop"]

        protected = data_df[str_protected]

        # Remove non-predictive, features with many missing values and target
        to_remove = (
            ["state", "county", "community", "communityname", "fold"]
            + list(data_df.columns[data_df.isnull().any()])
            + ["ViolentCrimesPerPop"]
            + str_protected
        )

        data_df.drop(columns=to_remove, inplace=True, axis="columns")

        X = data_df.to_numpy()
        y = target.to_numpy()
        p = protected.to_numpy()

        return X,y,p

    elif name == "ACSTravelTime":
        from folktables import travel_time_filter

        ACSTravelTimeCont = folktables.BasicProblem(
        features=[
            'AGEP',
            'SCHL',
            'MAR',
            'SEX',
            'DIS',
            'ESP',
            'MIG',
            'RELP',
            'RAC1P',
            'CIT',
            'OCCP',
            'JWTR',
            'POVPIP',
        ],
        target="JWMNP",
        group='AGEP',
        preprocess=travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
        )

        data_source = folktables.ACSDataSource(
            survey_year="2014", horizon="1-Year", survey="person"
        )

        acs_data = data_source.get_data(states=["MT"], download=True) # Use state Montana (MT)

        acs_data = acs_data.dropna(subset=['JWMNP']) # remove na rows from target

        features, target, protected = ACSTravelTimeCont.df_to_pandas(acs_data)

        to_encode = ["MAR","ESP","MIG","RELP","RAC1P","CIT","OCCP","JWTR"]
        features = pd.get_dummies(features, columns = to_encode,dtype=int)

        normalized_features = (features-features.min())/(features.max()-features.min())
        normalized_target = (target-target.min())/(target.max()-target.min())
        normalized_protected = (protected-protected.min())/(protected.max()-protected.min())

        return normalized_features.to_numpy(), normalized_target.to_numpy().reshape((-1,)),normalized_protected.to_numpy().reshape((-1,))

    elif name == "ACSIncome":

        ACSIncomeReg = folktables.BasicProblem(
            features=[
                "COW",
                "SCHL",
                "MAR",
                "OCCP",
                "RELP",
                "WKHP",
                "SEX",
                "RAC1P",
            ],
            target="PINCP",
            group="AGEP",
            preprocess=folktables.adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )

        data_source = folktables.ACSDataSource(
            survey_year="2014", horizon="1-Year", survey="person"
        )

        acs_data = data_source.get_data(states=["MT"], download=True)

        features, target, protected = ACSIncomeReg.df_to_pandas(acs_data)

        to_encode = ["COW", "MAR","OCCP","RELP","RAC1P"]
        features = pd.get_dummies(features, columns = to_encode,dtype=int)

        normalized_features = (features-features.min())/(features.max()-features.min())
        normalized_target = (target-target.min())/(target.max()-target.min())
        normalized_protected = (protected-protected.min())/(protected.max()-protected.min())

        return normalized_features.to_numpy(), normalized_target.to_numpy().reshape((-1,)),normalized_protected.to_numpy().reshape((-1,))

    else:
        pass


if __name__ == "__main__":
    load_dataset("ACSIncome")
