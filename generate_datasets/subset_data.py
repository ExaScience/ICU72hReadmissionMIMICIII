# Author: T.J.Ashby

import sys, yaml
import logging as lg
from pathlib import Path

import pandas as pd
import numpy as np


writeOpts = {"index": False}


def splitByPercentage(df_in, perc, seed=42):

    df = df_in.copy()

    df.loc[:, "Subset"] = "B"

    frac = perc / 100

    df.loc[df.sample(frac=frac, random_state=seed).index, "Subset"] = "A"

    return df


def splitByEthnicity(df_in, other="ALL"):

    df = df_in.copy()
    df.loc[:, "Subset"] = "Remove"

    def white(e):
        # print(e[0:5])
        return e[0:5] == "WHITE"

    df.loc[df.loc[:, "ETHNICITY"].map(white) == True, "Subset"] = "A"
 
    if other == "ALL":
        df.loc[df.loc[:, "ETHNICITY"].map(white) == False, "Subset"] = "B"
        assert (df["Subset"] == "Remove").any() == False
    else:
        df.loc[df.loc[:, "ETHNICITY"].map(lambda x: other in x), "Subset"] = "B"
        df = df.query("Subset != 'Remove'")

    return df


def splitByInsurance(df_in):

    df = df_in.copy()

    def insured(e):
        return e == "Private"

    df.loc[df.loc[:, "INSURANCE"].map(insured) == True, "Subset"] = "A"
    df.loc[df.loc[:, "INSURANCE"].map(insured) == False, "Subset"] = "B"

    return df


def splitBySystem(df_in, df_subj2sys, sys="CareVue"):

    df = df_in.copy()

    # Remove people in both systems
    n = len(df_subj2sys)
    df_subj2sys = df_subj2sys.query("~((MetaVision == True) & (CareVue == True))")
    lg.debug("Dropped {} subjects that cross systems".format(n - len(df_subj2sys)))

    # Check all subjects are in at least one of the systems
    df_check = df_subj2sys.query("(MetaVision == False) & (CareVue == False)")
    assert len(df_check) < 1

    # Get the overlap
    toGrab = set(df_subj2sys["SUBJECT_ID"]) & set(df["SUBJECT_ID"])
    lg.debug("Size of common set: {}".format(len(toGrab)))
    lg.debug(
        "Size of symmetric difference: {}".format(
            len(set(df_subj2sys["SUBJECT_ID"]) ^ set(df["SUBJECT_ID"]))
        )
    )

    # Select the subset based on system type (for overlap)
    df_subj2sys = df_subj2sys.set_index("SUBJECT_ID").loc[toGrab, :].reset_index()
    subj_A = df_subj2sys.loc[df_subj2sys[sys] == True, "SUBJECT_ID"]
    subj_B = df_subj2sys.loc[df_subj2sys[sys] == False, "SUBJECT_ID"]

    df = df.set_index("SUBJECT_ID")

    df.loc[subj_A, "Subset"] = "A"
    df.loc[subj_B, "Subset"] = "B"

    return df.reset_index()


#
# Demographic things it is possible to split on:
# - Age (via DoB), gender, insurance type, language, Religion,
#   Marital status, Ethnicity
#
def demographicInfo(df):

    recs = len(df)

    for c in [
        "INSURANCE",
        "LANGUAGE",
        "RELIGION",
        "MARITAL_STATUS",
        "ETHNICITY",
        "GENDER",
        "age",
    ]:
        lg.debug("Column {}".format(c))

        vals = list(df.loc[:, c].unique())

        percs = {}
        for v in vals:
            percs[v] = ((df.loc[:, c] == v).sum() / recs) * 100

        for k, v in sorted(percs.items(), key=lambda item: item[1]):
            lg.debug("  {}: {:.2f}%".format(k, v))

    return


def main(ymlfile):

    lg.basicConfig(stream=sys.stdout, level=lg.DEBUG)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", 20)
    pd.set_option("mode.chained_assignment", "raise")

    with open(ymlfile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    datadir = Path(config["datadir"])

    df_orig = pd.read_csv(datadir / "df_MASTER_DATA.csv")

    demographicInfo(df_orig)

    lg.debug(df_orig.head())

    totalRecs = len(df_orig)

    lg.debug("Records: {}".format(totalRecs))

    todo = ["Ethnicity", "Small", "Insurance", "System"]

    if "Ethnicity" in todo:
        lg.info("Split by Ethnicity")

        df = splitByEthnicity(df_orig)
        df_a = df.loc[df["Subset"] == "A", :]
        aRecs = len(df_a)

        lg.debug(
            "Subset A records: {} ({:.2f}%)".format(aRecs, (aRecs / totalRecs) * 100)
        )

        #
        # Write out subset A and B
        #
        df.to_csv(datadir / "df_MASTER_DATA_ethnicityWhite.csv", **writeOpts)

    if "Small" in todo:

        lg.info("Split by Percentage")

        # Loop over multiple percentages?
        df = splitByPercentage(df_orig, 1)
        df_a = df.loc[df["Subset"] == "A", :]
        aRecs = len(df_a)

        lg.debug(
            "Subset A records: {} ({:.2f}%)".format(aRecs, (aRecs / totalRecs) * 100)
        )

        #
        # Write out subset A and B
        #
        df_a.to_csv(datadir / "df_SMALL_DATA.csv", **writeOpts)

    if "Insurance" in todo:
        lg.info("Split by Insurance")

        df = splitByInsurance(df_orig)
        df_a = df.loc[df["Subset"] == "A", :]
        aRecs = len(df_a)

        lg.debug(
            "Subset A records: {} ({:.2f}%)".format(aRecs, (aRecs / totalRecs) * 100)
        )

        #
        # Write out subset A and B
        #
        df.to_csv(datadir / "df_MASTER_DATA_insuranceInsured.csv", **writeOpts)

    if "System" in todo:
        lg.info("Split by System")

        lg.info("CareVue")
        df = splitBySystem(df_orig, "CareVue")
        df_a = df.loc[df["Subset"] == "A", :]
        aRecs = len(df_a)

        lg.debug(
            "Subset A records: {} ({:.2f}%)".format(aRecs, (aRecs / totalRecs) * 100)
        )

        #
        # Write out subset A and B
        #
        df.to_csv(datadir / "df_MASTER_DATA_systemCV.csv", **writeOpts)

        lg.info("MetaVision")
        df = splitBySystem(df_orig, "MetaVision")
        df_a = df.loc[df["Subset"] == "A", :]
        aRecs = len(df_a)

        lg.debug(
            "Subset A records: {} ({:.2f}%)".format(aRecs, (aRecs / totalRecs) * 100)
        )

        #
        # Write out subset A and B
        #
        df.to_csv(datadir / "df_MASTER_DATA_systemMV.csv", **writeOpts)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("experiments.yml")
