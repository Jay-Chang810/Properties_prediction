import numpy as np
import pandas as pd
import pubchempy as pcp
from rdkit.Chem import MACCSkeys, AllChem, rdmolops
from rdkit import Chem
from keras.preprocessing import sequence
import subfunc_1 as subs

SMILES_0 = list(pd.read_csv('./DataPool/Compounds.csv', index_col=None, header=None)[0])

# encryption
smile_1 = np.zeros((len(SMILES_0), 50))

for i in range(len(SMILES_0)):
        print(SMILES_0[i])
        smile_1[i] = sequence.pad_sequences([subs.smile_parser(SMILES_0[i])], maxlen = 50)

pd.DataFrame(smile_1).to_csv('./DataPool/smiless.csv', index=None, header=None)

# build MACCS fp
maccs = np.zeros((len(SMILES_0), 167))
for i in range(len(SMILES_0)):
    print(SMILES_0[i])
    mol = Chem.MolFromSmiles(SMILES_0[i])
    fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()
    for j in range(len(fp)):
        maccs[i, j] = fp[j]
maccs = maccs[:, 1:]
pd.DataFrame(maccs).to_csv('./DataPool/MACCS.csv', index=None, header=None)

#build circular fp
cir = np.zeros((len(SMILES_0), 512))
for i in range(len(SMILES_0)):
    print(SMILES_0[i])
    mol = Chem.MolFromSmiles(SMILES_0[i])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=512).ToBitString()
    for j in range(len(fp)):
        cir[i, j] = fp[j]
cir = cir[:, :]
pd.DataFrame(cir).to_csv('./DataPool/Morgan512.csv', index=None, header=None)

#build topological fp
topo = np.zeros((len(SMILES_0), 2048))
for i in range(len(SMILES_0)):
    print(SMILES_0[i])
    mol = Chem.MolFromSmiles(SMILES_0[i])
    fp = rdmolops.RDKFingerprint(mol, fpSize=2048, minPath=1, maxPath=7).ToBitString()
    for j in range(len(fp)):
        topo[i, j] = fp[j]
topo = topo[:, :]
pd.DataFrame(topo).to_csv('./DataPool/TOPO2048.csv', index=None, header=None)