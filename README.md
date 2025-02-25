# XL-MSDigger
Here, we constructed Deep4D-XL, a deep learning tool capable of accurately predicting cross-linked peptide’s multi-dimensional information, including retention time, collisional cross-section, fragment ion intensity. Using Deep4D-XL as the core, we developed XL-MSDigger, a pipeline for comprehensive analysis of cross-linking mass spectrometry data acquired through both DDA and DIA approaches.
# Functions of XL-MSDigger
## Rescoring of DDA XL-MS data
The peptide list should be stored in a comma-separated values (CSV) file including two column:'Peptide', 'Charge'. This CSV file should be stored at the directory 'Deep4D/dataset/data/peptide_list.csv'
```
python XL-MSDigger_DDA.py --plinkfile plinkfile_dir --mgf_dir mgffile_dir --rescore_model 'dnn'
```
## DIA XL-MS analysis 
```
python XL-MSDigger_DIA.py --diann_report report_dir --DIA_library spectral_library_dir
```
## Building spectral library  

## Contacts
Please report any problems directly to the github issue tracker. Also, you can send feedback to moran_chen123@qq.com.
