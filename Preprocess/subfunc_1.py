import pyparsing as pp
import numpy as np
import os
from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from mpl_toolkits.mplot3d import Axes3D
from adjustText import adjust_text
import glob as gb
from keras.preprocessing import sequence


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def export_csv(Name,compound, y, y_pred, r2, run): #x,x_pred,
    createFolder('./CSV')
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    compound_ = np.asarray(compound).reshape(-1, 1)
    #act = np.hstack([compound_, x, y])
    #pred = np.hstack([compound_, x_pred, y_pred])
    act = np.hstack([compound_, r2.reshape([-1, 1]), y, y_pred])
    #column_pred = ['Name']+['Pred_Structure' + str(c) for c in range(30)]+['Pred_Var'+str(c) for c in range(51)]
    #column_act = ['Name']+['True_Structure' + str(c) for c in range(30)]+['True_Var' + str(c) for c in range(51)]
    column_act = ['Name'] + ['R2'] + ['True_Var' + str(c) for c in range(51)]+  ['Pred_Var' + str(c) for c in range(51)]
    pd.DataFrame(act, columns=column_act).to_csv('./CSV/CSV_act_' + Name +'_run' + str(run) + '.csv')
    #pd.DataFrame(pred, columns=column_pred).to_csv('./CSV/CSV_pred_' + Name + '.csv')

def smile_parser(str_smile):
    # str='FC(c1ccc(cc1)Cl)(F)F'

    atomicIndex = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                   'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
                   'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
                   'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
                   'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
                   'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
                   'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
                   'Tb': 65, 'Ty': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
                   'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
                   'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
                   'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                   'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
                   'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116,
                   'Ts': 117, 'Og': 118, 'c': 119, 'o': 120, 'n': 121, 's': 122,
                   '=': 123, '#': 124, '@': 125, '(': 126, ')': 127, '[': 128, ']': 129, '+': 130, '-': 139,
                   '0': 140, '1': 141, '2': 142, '3': 143, '4': 144, '5': 145, '6': 146, '7': 147, '8': 148, '9': 149,
                   '10': 150, '11': 151, '12': 152, '13': 153, '14': 154, '15': 155, '16': 156, '17': 157, '18': 158,
                   '19': 159, '20': 160, '21': 161, '23': 162, '24': 163, '31': 164, '32': 165, '34': 166, '35': 167,
                   '41': 168, '42': 169, '43': 170, '25': 171, '22': 172, '45': 173, '123': 174, '124': 175, '132': 176,
                   '134': 177, '134': 178, '135': 179, '125': 180, '33': 181, '234': 182}

    # Grammar definition
    isotope = pp.Regex('[1-9][0-9]?[0-9]?')
    atomclass = pp.Regex(':[0-9]+')
    bond = pp.oneOf(['-', '=', '#', '$', ':', '/', '\\', '.'])
    organicsymbol = pp.oneOf(['B', 'Br', 'C', 'Cl', 'N', 'O', 'P', 'S', 'F', 'I'])
    aromaticsymbol = pp.oneOf(['b', 'c', 'n', 'o', 'p', 's'])
    elementsymbol = pp.oneOf(['Al', 'Am', 'Sb', 'Ar', 'At', 'Ba', 'Bk', 'Be', 'Bi',
                              'Bh', 'B', 'Br', 'Cd', 'Ca', 'Cf', 'C', 'Ce', 'Cs', 'Cl', 'Cr',
                              'Co', 'Cu', 'Cm', 'Ds', 'Db', 'Dy', 'Es', 'Er', 'Eu', 'Fm',
                              'F', 'Fr', 'Gd', 'Ga', 'Ge', 'Au', 'Hf', 'Hs', 'He', 'Ho', 'H',
                              'In', 'I', 'Ir', 'Fe', 'Kr', 'La', 'Lr', 'Pb', 'Li', 'Lu', 'Mg',
                              'Mn', 'Mt', 'Md', 'Hg', 'Mo', 'Nd', 'Ne', 'Np', 'Ni', 'Nb', 'N',
                              'No', 'Os', 'O', 'Pd', 'P', 'Pt', 'Pu', 'Po', 'K', 'Pr', 'Pm',
                              'Pa', 'Ra', 'Rn', 'Re', 'Rh', 'Rg', 'Rb', 'Ru', 'Rf', 'Sm',
                              'Sc', 'Sg', 'Se', 'Si', 'Ag', 'Na', 'Sr', 'S', 'Ta', 'Tc',
                              'Te', 'Tb', 'Tl', 'Th', 'Tm', 'Sn', 'Ti', 'W', 'Uub', 'Uuh',
                              'Uuo', 'Uup', 'Uuq', 'Uus', 'Uut', 'Uuu', 'U', 'V', 'Xe', 'Yb',
                              'Y', 'Zn', 'Zr'])

    integer = pp.Word("0123456789")
    ringclosure = pp.Optional(pp.Literal('%') + pp.oneOf(['1 2 3 4 5 6 7 8 9'])) + pp.oneOf(['1 2 3 4 5 6 7 8 9'])
    charge = (pp.Literal('-') + pp.Optional(
        pp.oneOf(['-02-9']) ^ pp.Literal('1') + pp.Optional(pp.oneOf(['0-5'])))) ^ pp.Literal('+') + pp.Optional(
        pp.oneOf(['+02-9']) ^ pp.Literal('1') + pp.Optional(pp.oneOf('[0-5]')))
    chiralclass = pp.Optional(
        pp.Literal('@') + pp.Optional(pp.Literal('@')) ^ (pp.Literal('TH') ^ pp.Literal('AL')) + pp.oneOf(
            '[1-2]') ^ pp.Literal('SP') + pp.oneOf('[1-3]') ^ pp.Literal('TB') + (
                    pp.Literal('1') + pp.Optional(pp.oneOf('[0-9]')) ^ pp.Literal('2') + pp.Optional(
                pp.Literal('0')) ^ pp.oneOf('[3-9]')) ^ pp.Literal('OH') + (
                    (pp.Literal('1') ^ pp.Literal('2')) + pp.Optional(pp.oneOf('[0-9]')) ^ pp.Literal(
                '3') + pp.Optional(pp.Literal('0')) ^ pp.oneOf('[4-9]')))

    atomspec = pp.Literal('[') + pp.OneOrMore(pp.Optional(isotope) + (
                pp.Literal('se') ^ pp.Literal('as') ^ aromaticsymbol ^ elementsymbol ^ pp.Literal('*'))+  pp.Optional(
        chiralclass) + pp.Optional(integer) + pp.Optional(charge) + pp.Optional(atomclass)) + pp.Literal(']')

    atom = (organicsymbol + pp.Optional(integer)) ^ (aromaticsymbol + pp.Optional(integer)) ^ pp.Literal('*') ^ (
                atomspec + pp.Optional(integer))
    chain = pp.OneOrMore(pp.Optional(bond) + (atom ^ ringclosure))
    smiles = pp.Forward()
    branch = pp.Forward()

    smiles << atom + pp.ZeroOrMore(chain ^ branch)
    branch << (pp.Literal('(') + (pp.OneOrMore(bond + pp.OneOrMore(smiles))^pp.OneOrMore(smiles)) + pp.Literal(')'))

    formulaData = smiles.parseString(str_smile)
    #print('specie:' + str_smile.replace(" ", ""))
    if len(str_smile.replace(" ", ""))!=len(''.join(str(e) for e in formulaData)):
        print ('wrong:'+ str_smile.replace(" ", "") + ' => ' + ''.join(str(e) for e in formulaData) + '  &act_len:' + str(len(str_smile.replace(" ", "")))+ ' &trans_len:' +str(len(''.join(str(e) for e in formulaData))))

    formulaData_Index = [atomicIndex[c] for c in formulaData]

    return formulaData_Index

def loaddata(path='DataPool', type='train', len_compound=50):
    file = pd.read_csv('./' + path + '/' + type + '_dataset.csv', index_col=None, header=None)
    dataset = np.array(file.values)
    compound = dataset[:,0].tolist()
    num_compound = len(compound)
    compound_index = np.zeros([num_compound, len_compound])
    sigma_data = np.zeros([num_compound, 51, 2])
    # pad smiles file into array
    for n in range(num_compound):
        formulaData_Index = smile_parser(compound[n])
        compound_index[n] = sequence.pad_sequences([formulaData_Index], maxlen=len_compound)
        sigma_data[n] = dataset[n, 1:].reshape([2, 51]).T
    return num_compound, compound, compound_index, sigma_data[:, :, 1]

def dataParser(file,data_split = 0.5,seed=20,Train_data_selected_list=[]):
    caps = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lowers = caps.lower()
    digits = "0123456789"
    symbolic = "~!@#$%^&*()-_+{}[].="
    smile = pp.Literal("#") + pp.Optional(" ") + pp.Word(caps + lowers + digits + symbolic)
    sigma = pp.Word(symbolic + digits) + pp.Optional(" ") + pp.Word(symbolic + digits)
    data_parser = pp.ZeroOrMore(sigma ^ smile)

    compound = [] #* Num_compound
    sigma_data = np.zeros([2000, 51, 2])
    fp = open(file, "r")  # 讀檔
    compound_index = -1
    fileStringx = fp.read()
    fp.close()
    fileString_split=fileStringx.split("\n")
    count = len(fileString_split)
    for n in range(count):
        #fileString = fp.readline()  # 把整個檔案讀入buffer
        fileString = fileString_split[n]
        # print (fileString)
        data_structure = data_parser.parseString(fileString)
        # print(data_structure)
        if not data_structure:
            continue
        elif data_structure[0] == "#":
            compound_index += 1
            sigma_index = 0
            compound.append(data_structure[1])
        else:
            data_structure = [float(i) for i in data_structure]
            sigma_data[compound_index, sigma_index, :] = data_structure
            sigma_index += 1
    if os.path.isdir("./DataPool"):
        import shutil
        shutil.rmtree("./DataPool")
    createFolder("./DataPool")
    shuffle_index = np.arange(len(compound), dtype=np.int32)
    np.random.seed(seed)
    np.random.shuffle(shuffle_index)
    compound = np.asarray(compound)[shuffle_index].tolist()
    sigma_data = sigma_data[shuffle_index, :, :]
    Num_compound = len(compound)
    data_split_bound = int(Num_compound * data_split)

    Num_Train=0
    Num_Test=0
    Train_list=[]
    Train_dataset = np.empty([data_split_bound,sigma_data.shape[1]*2+1],dtype='<U200')
    Test_dataset = np.empty([Num_compound-data_split_bound,sigma_data.shape[1]*2+1],dtype='<U200')

    #selected list
    for i in range(len(compound)):
        if compound[i] in Train_data_selected_list:
            Train_list.append(i)
            Train_dataset[Num_Train] = np.hstack([compound[i], sigma_data[i].T.reshape([-1])])
            Num_Train+=1
    #Rest Train/Test data
    for i in range(len(compound)):
        if (i in Train_list):
            print("conflict:" + compound[i])
        if not (i in Train_list):
            if (Num_Train<data_split_bound):
                Train_list.append(i)
                Train_dataset[Num_Train] = np.hstack([compound[i], sigma_data[i].T.reshape([-1])])
                Num_Train += 1
            else:
                Test_dataset[Num_Test] = np.hstack([compound[i], sigma_data[i].T.reshape([-1])])
                Num_Test += 1
                #print("Train done")


    pd.DataFrame(Train_dataset).to_csv('./DataPool/train_dataset.csv', header=False, index=False)
    pd.DataFrame(Test_dataset).to_csv('./DataPool/test_dataset.csv', header=False, index=False)