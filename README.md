Source code for TKDE submission
DRLPG: Reinforced Opponent-aware Order Pricing for Hub Mobility Services

1. Step1:
    - func1-func9.py the regressors to simulate quotations.
    - train_quofuncs.py the training and generating progresses.
2. Step2:
    - convlstm.py convlstm module.
    - dataloader.py customized dataloader from pytorch.
    - predic_model.py the whole quotation prediction model.
    - train_3.py training the whole quotation prediction model.
    - triditionalmodel.py linear and poly. regression.
    - eval.py evaluation file.
3. Step3:
    - evalpred.py eval file for preditions.
    - extract.py for tensorboard.
    - heurresult.py for tensorboard and result.
    - utils.py environment utils.
    - heuristic_agent.py agents file.
    - heurtest.py testing file.
4. dataset:
    The data from the real-world hub mobility service provider.

