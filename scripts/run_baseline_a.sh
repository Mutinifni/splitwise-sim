python run.py \
    applications.0.scheduler=token_jsq \
    cluster=half_half \
    cluster.servers.0.count=40 \
    cluster.servers.1.count=0 \
    start_state=baseline \
    performance_model=db \
    trace.filename=rr_conversation_80 \
    seed=0
    #+experiment=traces_light \
