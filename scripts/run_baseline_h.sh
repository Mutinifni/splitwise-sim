python run.py \
    applications.0.scheduler=token_jsq \
    cluster=half_half \
    cluster.servers.0.count=0 \
    cluster.servers.1.count=40 \
    start_state=baseline \
    performance_model=db \
    trace.filename=rr_conv_80 \
    seed=0
    #+experiment=traces_light \
