from pipeline.sgu_trainer import run_sgu_training
from pipeline.agent_trainer import run_agent_training_pipeline

def execute(symbol="688981", PHI=0.001, TICK_SIZE=0.01, fee_rate=0.0003, use_fee=False, use_arl=False):

    run_sgu_training(
        symbol=symbol,
        train_range=(20240401, 20240528), 
        val_range=(20240529, 20240612),
    )

    run_agent_training_pipeline(
        symbol=symbol,
        sgu_train_range=(20240401, 20240528), 
        PHI=PHI,
        TICK_SIZE=TICK_SIZE,
        fee_rate=fee_rate,
        USE_FEE=use_fee,
        USE_ARL=use_arl
    )


if __name__ == '__main__':
    # execute(symbol='688981', PHI=0.005, TICK_SIZE=0.01, fee_rate=0.0, use_fee=False, use_arl=False)
    execute(symbol='510300', PHI=0.0001, TICK_SIZE=0.001, fee_rate=0.00005, use_fee=False, use_arl=True)