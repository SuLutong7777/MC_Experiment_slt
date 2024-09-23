from dataset_real2 import Real_Data
train_json = '/home/sulutong/mr2nerf-master/colmap2nerf/Real_World_Trans_custom/stump/transforms_train.json'
val_json = '/home/sulutong/mr2nerf-master/colmap2nerf/Real_World_Trans_custom/stump/transforms_val.json'
test_json = '/home/sulutong/mr2nerf-master/colmap2nerf/Real_World_Trans_custom/stump/transforms_test.json'


if __name__ == "__main__":
    real_data = Real_Data(train_json, test_json, val_json)
