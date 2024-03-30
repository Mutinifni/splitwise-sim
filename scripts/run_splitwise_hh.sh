python run.py \
    applications.0.scheduler=mixed_pool \
    applications.0.model_architecture=llama2-70b \
    applications.0.model_size=llama2-70b-fp16 \
    cluster=half_half \
    cluster.servers.0.count=1 \
    cluster.servers.1.count=1 \
    start_state=splitwise \
    start_state.prompt.num_instances=1 \
    start_state.token.num_instances=1 \
    performance_model=db \
    trace.filename=rr_code_2 \
    debug=True \
    seed=0
    #applications.0.scheduler=token_jsq \
    #trace.filename=rr_code_70 \

