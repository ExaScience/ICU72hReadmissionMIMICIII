# Author: T.J. Ashby

import sys, os, subprocess
from pathlib import Path
import logging as lg
import yaml

import pandas as pd
import dask
from dask import distributed

import subset_data
import fold_saver


def doOriginal(client, datadir, out):

    dfname = "df_MASTER_DATA.csv"

    lg.info("Launch experiment {}".format(dfname))

    outdir = out / "results_original"
    os.makedirs(outdir, exist_ok=True)

    fut = client.submit(fold_saver.main, *[dfname, datadir, outdir])

    lg.info("Experiment submitted")

    return fut


def doEthnicity(client, datadir, out):

    dfname = "df_MASTER_DATA_ethnicityWhite.csv"

    #
    # Prepare the data
    #
    if not (datadir / dfname).exists():
        lg.info("Generate data for {}".format(dfname))
        dforigname = "df_MASTER_DATA"
        df_orig = pd.read_csv(datadir / f"{dforigname}.csv")
        df = subset_data.splitByEthnicity(df_orig)

        lg.info("Writing out data for {}".format(dfname))
        df.to_csv(datadir / dfname, index=False)
    else:
        lg.info(f"{dfname} exits, reading it")
        df = pd.read_csv(datadir / dfname)

    lg.info("Launch experiment {}".format(dfname))

    outdir = out / "subset_white"
    os.makedirs(outdir, exist_ok=True)

    fut = client.submit(fold_saver.main, *[dfname, datadir, outdir])

    lg.info("Experiment submitted")

    return fut


def doInsurance(client, datadir, out):

    dfname = "df_MASTER_DATA_insuranceInsured.csv"

    #
    # Prepare the data
    #
    if not (datadir / dfname).exists():
        lg.info("Generate data for {}".format(dfname))
        dforigname = "df_MASTER_DATA"
        df_orig = pd.read_csv(datadir / f"{dforigname}.csv")
        df = subset_data.splitByInsurance(df_orig)

        lg.info("Writing out data for {}".format(dfname))
        df.to_csv(datadir / dfname, index=False)
    else:
        lg.info(f"{dfname} exits, reading it")
        df = pd.read_csv(datadir / dfname)

    lg.info("Launch experiment {}".format(dfname))

    outdir = out / "subset_insured"
    os.makedirs(outdir, exist_ok=True)

    fut = client.submit(fold_saver.main, *[dfname, datadir, outdir])

    lg.info("Experiment submitted")

    return fut


def doPercs(client, datadir, out):

    fname = "df_MASTER_DATA"

    df_orig = pd.read_csv(datadir / f"{fname}.csv")

    # Do experiments with increasing fraction of training data
    percs = [0.5]
    percs += [1, 2, 4, 8, 16]
    percs += [20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]
    futs = []

    lg.info("Splits to run: {}".format(percs))

    for p in percs:
        noDot = str(p).replace(".", "-")
        dfname = f"{fname}_{noDot}percSubset.csv"

        #
        # Subsample the data set
        #
        if not (datadir / dfname).exists():
            lg.info("Generate data for {}".format(dfname))
            df = subset_data.splitByPercentage(df_orig, p)

            lg.info("Writing out data for {}".format(dfname))
            df.to_csv(datadir / dfname, index=False)
        else:
            lg.info(f"{dfname} exits, reading it")
            df = pd.read_csv(datadir / dfname)

        lg.info("Launch experiment {}".format(dfname))

        outdir = out / "perc_splits" / f"{fname}_{noDot}_perc"
        os.makedirs(outdir, exist_ok=True)

        fut = client.submit(fold_saver.main, *[dfname, datadir, outdir])

        lg.info("Experiment submitted")

        futs.append(fut)

    return futs


def doSystem(client, datadir, out):

    dfname = "df_MASTER_DATA_systemCV.csv"

    #
    # Prepare the data
    #
    if not (datadir / dfname).exists():
        lg.info("Generate data for {}".format(dfname))
        dforigname = "df_MASTER_DATA"
        df_orig = pd.read_csv(datadir / f"{dforigname}.csv")

        df_eventscv = pd.read_csv(datadir / "INPUTEVENTS_CV.csv.gz")
        df_eventsmv = pd.read_csv(datadir / "INPUTEVENTS_MV.csv.gz")
        subj_cv = list(df_eventscv["SUBJECT_ID"].unique())
        subj_mv = list(df_eventsmv["SUBJECT_ID"].unique())

        df_subj = pd.DataFrame(data=set(subj_cv) | set(subj_mv),
                               columns=["SUBJECT_ID"]).set_index("SUBJECT_ID")
        df_subj.loc[:, 'CareVue'] = False
        df_subj.loc[set(subj_cv), 'CareVue'] = True
        df_subj.loc[:, 'MetaVision'] = False
        df_subj.loc[set(subj_mv), 'MetaVision'] = True
        df_subj = df_subj.reset_index()

        df = subset_data.splitBySystem(df_orig, df_subj)

        lg.info("Writing out data for {}".format(dfname))
        df.to_csv(datadir / dfname, index=False)
    else:
        lg.info(f"{dfname} exits, reading it")
        df = pd.read_csv(datadir / dfname)

    lg.info("Launch experiment {}".format(dfname))

    outdir = out / "subset_systemCV"
    os.makedirs(outdir, exist_ok=True)

    fut = client.submit(fold_saver.main, *[dfname, datadir, outdir])

    lg.info("Experiment submitted")

    return fut


def setupCluster(nodes):

    try:

        # Remove the first node, so it can act as a head node without getting
        # overloaded and choking the parallel progress
        headnode = True
        if headnode:
            headNode = nodes[0]
            if len(nodes) > 1:
                nodes = nodes[1:]
                lg.info("Head node: {}".format(headNode))

        lg.info("Dask node(s): {}".format(nodes))

        #
        # Blindly trusting the hosts is a work-around as I can't figure out how
        # to do it properly. However, this is a SECURITY HOLE!
        #
        asyncssh_opts = {"known_hosts": None}
        cluster = distributed.deploy.ssh.SSHCluster(
            hosts=nodes, connect_options=asyncssh_opts
        )
        client = distributed.Client(cluster)

        lg.info("Distributed cluster set up with {} worker nodes".format(len(nodes)))

    except Exception as err:
        lg.warning("Couldn't set up cluster: {}".format(err))
        lg.warning("Are you running on the head node?")
        client = distributed.Client(n_workers=4, threads_per_worker=1)
        lg.info("Running locally")

    return client


def teardownCluster(client):

    lg.info("Tearing down cluster...")

    try:
        client.shutdown()
        # cluster.close()

    except BaseException as err:
        lg.warning("Couldn't tear down cluster: {}".format(err))

    lg.info("...teardown finished")

    return


def doDask(nodes, config):

    # Extract info from the yaml config
    experiments = config["experiments"]
    datadir = Path(config["datadir"])
    out = Path(config["out"])
    perc_out = Path(config["perc_out"])
    

    client = setupCluster(nodes)
    futs = []

    if "org" in experiments:
        futs.append(doOriginal(client, datadir, out))

    if "ethn" in experiments:
        futs.append(doEthnicity(client, datadir, out))

    if "sys" in experiments:
        futs.append(doSystem(client, datadir, out))

    if "percs" in experiments:
        futs += doPercs(client, datadir, perc_out)


    # Actually do the work
    lg.info("Trigger computations...")
    results = client.gather(futs)
    lg.info("...computations done!")
    lg.info(results)

    teardownCluster(client)


def main(ymlfile):

    #
    # Set-up the output
    #
    lg.basicConfig(stream=sys.stderr, level=lg.DEBUG)

    noisy = ["asyncssh", "asyncio", "distributed.deploy.ssh"]
    for n in noisy:
        logger = lg.getLogger(n)
        logger.setLevel(lg.WARNING)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", 20)

    pbs = "PBS_NODEFILE"
    if pbs in os.environ:
        nf = open(os.environ["PBS_NODEFILE"])
        nodes = [n.strip() for n in nf.readlines()]
    else:
        nodes = []

    with open(ymlfile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lg.debug(config)

    doDask(nodes, config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("experiments.yml")
