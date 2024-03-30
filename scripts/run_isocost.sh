python run.py --multirun applications.0.scheduler=mixed_pool +experiment=splitwise_hh_isocost seed=0 &
python run.py --multirun applications.0.scheduler=mixed_pool +experiment=splitwise_aa_isocost seed=0 &
python run.py --multirun applications.0.scheduler=mixed_pool +experiment=splitwise_ha_isocost seed=0 &
python run.py --multirun applications.0.scheduler=mixed_pool +experiment=splitwise_hhcap_isocost seed=0
