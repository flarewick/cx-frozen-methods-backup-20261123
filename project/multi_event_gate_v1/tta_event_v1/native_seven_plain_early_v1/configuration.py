"""Fixed requested groups and denominator, never online predictor features."""
GROUPS={
    'plain':dict(tasks=('lift_pot','handover_mic','stack_bowls_two'),lanes=8,priority=('stack_bowls_two','handover_mic','lift_pot'),TRAIN_version='native_seven_plain_dev_v3',TRAIN_stage='dev_plain'),
    'gated':dict(tasks=('beat_block_hammer','turn_switch','adjust_bottle','open_laptop'),lanes=12,priority=('turn_switch','open_laptop','beat_block_hammer','adjust_bottle'),TRAIN_version='gated_postexit_recovery_v2',TRAIN_stage='qualified_TRAIN'),
}

def bounds(number):
    assert type(number)is int and 0<=number<9
    assert number==0
    return 0,1
