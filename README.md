# TranTCR: *Transformer Model for TCR-Epitope Interaction*

## Dependencies
See requirements.txt for all TranTCR dependencies.It is recommended to create a virtualenv and installing TranTCR within this environment to ensure proper versioning of dependencies.
## Installation
You could first set up a virtual environment for TranTCR using conda:
```bash
    conda create -n TranTCR python=3.8
    conda activate TranTCR
    # Install Pytorch packages (for CUDA 11.6)
    conda install pytorch==1.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
    pip install -r requirements.txt
    conda install -c bioconda anarci
```

## Inference
### Train
For TranTCR-bind:
```bash
    cd TranTCR-bind
    python train.py
```

### predict
For TranTCR-bind:
```bash
    cd TranTCR-bind
    python predict.py -t ./test_data/ImmuneCode/random_mutation/unique_cdr3_1V1_Immunecode.csv -o ./output/result_test.csv
```


## Contact
If you have any questions, please contact us at 230218235@seu.edu.cn