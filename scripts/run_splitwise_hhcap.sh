python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=hhcap_half_half \
    cluster.servers.0.count=5 \
    cluster.servers.1.count=35 \
    start_state=splitwise_hhcap \
    start_state.split_type=heterogeneous \
    trace.filename=rr_conversation_80 \
    performance_model=db \
    seed=0
