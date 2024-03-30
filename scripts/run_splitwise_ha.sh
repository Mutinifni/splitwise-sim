python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=half_half \
    cluster.servers.0.count=26 \
    cluster.servers.1.count=25 \
    start_state=splitwise \
    start_state.split_type=heterogeneous \
    performance_model=db \
    trace.filename=rr_conversation_80 \
    seed=0
    #applications.0.scheduler=token_jsq \
    #trace.filename=rr_code_70 \

