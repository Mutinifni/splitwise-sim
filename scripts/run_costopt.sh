python run.py --multirun seed=0 +experiment=baseline_h100_costopt &
python run.py --multirun seed=0 +experiment=baseline_a100_costopt
python run.py --multirun seed=0 applications.0.scheduler=mixed_pool +experiment=splitwise_hh_costopt &
python run.py --multirun seed=0 applications.0.scheduler=mixed_pool +experiment=splitwise_aa_costopt
python run.py --multirun seed=0 applications.0.scheduler=mixed_pool +experiment=splitwise_ha_costopt &
python run.py --multirun seed=0 applications.0.scheduler=mixed_pool +experiment=splitwise_hhcap_costopt
