python run.py --multirun \
    applications.0.scheduler=token_jsq \
    cluster=isocost_a100,isopower_a100,dgx-h100 \
    start_state=baseline \
    performance_model=db \
    seed=0 \
    +experiment=traces
    #cluster=dgx-h100,isocost_a100,isopower_a100,isocount_a100 \
    #applications.0.scheduler=round_robin \
    #trace.filename=rr_1,rr_2,rr_3,rr_4,rr_5,rr_6,rr_7,rr_8,rr_9,rr_10 \
