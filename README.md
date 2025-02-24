# XL-MSDigger
## Introduction
Here, we constructed Deep4D-XL, a deep learning tool capable of accurately predicting cross-linked peptide’s multi-dimensional information, including retention time, collisional cross-section, fragment ion intensity. Using Deep4D-XL as the core, we developed XL-MSDigger, a pipeline for comprehensive analysis of cross-linking mass spectrometry data acquired through both DDA and DIA approaches.
# Functions of XL-MSDigger
## Rescoring of DDA XL-MS data
### 1. Generate 4D library
The peptide list should be stored in a comma-separated values (CSV) file including two column:'Peptide', 'Charge'. This CSV file should be stored at the directory 'Deep4D/dataset/data/peptide_list.csv'
```
Peptide,Charge
AAAAAAAAGAFAGR,2
aAAAAAAAAVPSAAGR,2
AAAAAATAPPSPGPAQPGPR,2
AAAAALeSQQQSLQER,2
AAAAAWEEPSSGNGTAR,2
AAAAFVLsANENNIALFK,2
```
The PTM contain phosphorylation of serine, threonine and tyrosine, oxidation of methionine and acetylation on N-terminal of proteins. They are represented as: S(phos)--s, T(phos)--t, Y(phos)--y, M(oxid)--e, acetylation--a.
## License
Deep4D is distributed under an Apache License. See the LICENSE file for details.
## Contacts
Please report any problems directly to the github issue tracker. Also, you can send feedback to moran_chen123@qq.com.
