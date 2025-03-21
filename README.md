# XL-MSDigger
Here, we constructed Deep4D-XL, a deep learning tool capable of accurately predicting cross-linked peptide’s multi-dimensional information, including retention time, collisional cross-section, fragment ion intensity. Using Deep4D-XL as the core, we developed XL-MSDigger, a pipeline for comprehensive analysis of cross-linking mass spectrometry data acquired through both DDA and DIA approaches.
# Functions of XL-MSDigger
## Rescoring of DDA XL-MS data
The peptide list should be stored in a comma-separated values (CSV) file including two column:'Peptide', 'Charge'. This CSV file should be stored at the directory 'Deep4D/dataset/data/peptide_list.csv'
```
python XL-MSDigger_DDA.py --plinkfile './test_data/plink_test' --mgf_dir './test_data/test.mgf' --rescore_model 'dnn'
```
Description of argparse:  
--plinkfile: The file directory of pLink2 output.  
--mgf_dir: The file directory of mgf file.  
--rescore_model: You can select 'dnn' and 'svm'.  
## DIA XL-MS analysis 
```
python XL-MSDigger_DIA.py --diann_report './test_data/report.tsv' --DIA_library './test_data/library.csv'

```
Description of argparse:  
--diann_report: The file directory of DIA-NN report.  
--DIA_library: The file directory of DIA library.  
## Building spectral library  
```
python Build_library.py --experiment_library './test_data/experimental_library.csv' --aim_protein './test_data/aim_PPI.csv' --aim_type 1 --fasta_dir './test_data/human reviewed.fasta'
```
Description of argparse:  
--experiment_library: The file directory of experimental library.  
--aim_protein: The file directory of aim protein.  
--aim_type: The file directory of ccs model.  
--fasta_dir: The file directory of fasta file.  
--msms_param_dir: The file directory of msms model.     
--rt_param_dir: The file directory of rt model.  
--ccs_param_dir: The file directory of ccs model.     
--maxcharge: The maximum charge in peptide list, the charge range would in [3,maxcharge].  
--slice: If the scale of  peptide list is too large for your compute, you can slice the list. It should be set to an integer greater than 0.  
--batch_size: The batch size.  
## Contacts
Please report any problems directly to the github issue tracker. Also, you can send feedback to moran_chen123@qq.com.
