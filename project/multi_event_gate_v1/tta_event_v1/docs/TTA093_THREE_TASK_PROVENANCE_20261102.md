# cup / shoe / mouse 的 TTA093 来源核对

三个任务均复用blocks/bowls共享M_event_full残差adapter头，以及同一个响应checkpoint0937511a...，未用cup/shoe/mouse自己的TRAIN数据做Bθ方向监督。基础VLA冻结；在线仅更新每回合128维gain/offset快参数，两步SGD，局部自身视觉位移反馈，原事件选中时作用、原冷却/总3/trust，回合清零。

|任务|同场完整baseline|冻结full|TTA093|
|---|---:|---:|---:|
|place_empty_cup|该两路实验未评估|40/100|41/100|
|place_shoe|该两路实验未评估|65/100|67/100|
|place_mouse_pad|24/100|29/100|35/100|

cup/shoe的权威complete_comparison在outputs/native_cross_task100_v19，只含conditions full/TTA；不能据此声称full优于baseline。mouse来自outputs/native_frozen100_v25/mixed完整三路配对。三者TTA相对full净+1/+2/+6，均无反向损失景；cup/shoe差异小，不应夸大稳定性或显著性。mouse配对p=.03125未多重校正。

原离线方向训练只用blocks/bowls：136FIT、38VAL，监督反馈后第一个实际被原门控选中的动作。cup/shoe只扩展语义关系接口v1，mouse接口v2；权重、128维快参数、两步更新和原事件作用数学不变。

用户当前选择沿用这三项做法：不使用新增任务TRAIN专家标签校准Bθ；以原TTA093的权重、gain/offset与原事件作用时机为基准，只做必要的新对象/反馈接口扩展，并单独验证。V5额外非原事件作用及V6 offset-only不是该冻结方案，不能把其失败当成093在新任务的正式结论。

## 对此前CPU探针的口径更正

outputs/native_object_update_diagnostics_v2/CPU_original_VAL_meta_alignment.json使用早期通用PacketReader/反馈到达query损失，56包；它不是当前093真正的selected-action外层验证，不能用来评价093的38包正式VAL训练目标或推导需要新任务TRAIN。原记录保留但该解释撤回。真正训练器native_direction_visible_training_v1/train.py使用QualifiedSelectedReader和native_selected_outer_target_v1/loss.py。实际38VAL：step0=.033981777893，step1000=.034093343065，zero=.033924173668，均仅代理损失，不替代原生任务收益。

native_offset_response_v2草案的CPU算术一致/梯度有限检查仍是数学检查，其通用query外层目标不等于093正式目标；未训练、未部署、不作为下一正式候选依据。原70回合V6原生负结果及完整验收不受此更正影响。
