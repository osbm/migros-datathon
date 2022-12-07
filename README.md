
preparation


```
kaggle competitions download -c migros-datathon-coderspace
python -m venv .venv
.venv/bin/activate
pip install -r requirements.txt

```
> First run `preprocess_transaction_sale.py` to generate more useful version of the `transaction_sale.csv` and then `preprocess.py` to generate the final dataframes.

# plan

### Deal with imbalanced data

oversampling and undersampling
data augmentation

### Feature engineering

### Ensemble
VotingClassifier
