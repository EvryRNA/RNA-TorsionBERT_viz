LIST_ALL_METRICS = [
    "RMSD",
    r"$INF_{all}$",
    "DI",
    "MCQ",
    "GDT-TS",
    "CAD-score",
    "εRMSD",
    "TM-score",
    "lDDT",
    "MAE",
]
LIST_METRICS_WITHOUT_MAE = [
    "RMSD",
    r"$INF_{all}$",
    "DI",
    "MCQ",
    "GDT-TS",
    "CAD-score",
    "εRMSD",
    "TM-score",
    "lDDT",
    "MAE",
]
LIST_ENERGIES_WITH_MAE = ["RASP", "εSCORE", "DFIRE", "rsRNASP", "RNA-Torsion-A"]
LIST_ENERGIES = ["RASP", "εSCORE", "DFIRE", "rsRNASP"]
NEGATIVE_ENERGY = ["εSCORE"]

ASCENDING_METRICS = [
    "CAD-score",
    r"$INF_{all}$",
    "INF-WC",
    "INF-NWC",
    "INF-STACK",
    "GDT-TS",
    "TM-SCORE",
    "GDT-TS@1",
    "GDT-TS@2",
    "GDT-TS@4",
    "GDT-TS@8",
    "lDDT",
    "TM-score",
]
ASCENDING_ENERGIES = ["BARNABA-eSCORE", "RNA3DCNN", "εSCORE"]

DICT_TO_CHANGE = {
    "BARNABA-eRMSD": "εRMSD",
    "BARNABA-eSCORE": "εSCORE",
    "RASP-ENERGY": "RASP",
    "tm-score-ost": "TM-score",
    "lddt": "lDDT",
    "RNA3DCNN_MDMC": "RNA3DCNN",
    "RNA3DCNN_MD": "RNA3DCNN",
    "INF-ALL": r"$INF_{all}$",
    "CAD": "CAD-score",
    "mae_gold": "MAE",
    "mae_dna": "RNA-Torsion-A",
}
