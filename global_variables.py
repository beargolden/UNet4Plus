# Define global variables

# BATCH_SIZE: 一次训练所抓取的数据样本数量
BATCH_SIZE = 32

# DATASET_NAME: 数据集名称 (BickleyDiary, DIBCO, PLM)
DATASET_NAME = "DIBCO"

# NETWORK_MODEL: 深度网络模型名称 (UNet, UNet1Plus_w_DeepSupv, UNet1Plus_wo_DeepSupv, UNet2Plus_w_DeepSupv, UNet2Plus_wo_DeepSupv,
# UNet3Plus_w_DeepSupv, UNet3Plus_wo_DeepSupv, UNet4Plus_w_DeepSupv, UNet4Plus_wo_DeepSupv)
NETWORK_MODEL = "UNet4Plus_wo_DeepSupv"

# LOSS_FUNCTION: 深度网络损失函数 (BCE_Dice, BCE_Dice_mIoU)
LOSS_FUNCTION = "BCE_Dice"

# NUM_EPOCHS: 最大训练迭代次数
NUM_EPOCHS = 500

# NUM_FILTERS: 网络第一层通道滤波器数量
NUM_FILTERS = 32

# TILE_SIZE: 图像子块大小
TILE_SIZE = 128
