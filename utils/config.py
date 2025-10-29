class Config:
    def __init__(self,task):
        if "la" in task: # SSL
            self.base_dir = './Datasets/LASeg'
            self.save_dir = './LA_data'
            self.patch_size = (112, 112, 80)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 16
            self.early_stop_patience = 40
            self.batch_size = 4

        elif "brats2019" in task:
            self.base_dir = './Datasets/BraTS2019/'
            self.save_dir = './Brats2019_data'
            self.patch_size = (96, 96, 96)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 16
            self.early_stop_patience = 40
            self.batch_size = 4

        else:
            raise NameError("Please provide correct task name, see config.py")


