python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=half_half \
    cluster.servers.0.count=40 \
    cluster.servers.1.count=0 \
    start_state=splitwise \
    start_state.prompt.num_instances=27 \
    start_state.token.num_instances=13 \
    performance_model=db \
    trace.filename=rr_conv_80 \
    seed=0
    #applications.0.scheduler=token_jsq \
    #trace.filename=rr_code_70 \
