
class cfg:
    def __init__(self, num_epochs, learning_rate, use_focal, use_wandb, save_file, run_name):
        """ Init configuration object

        Args:
            num_epochs (int): number of epochs for training.
            learning_rate (float): learning rate.
            use_focal (bool): use focal loss or not.
            use_wandb (bool): use WandB or not.
            save_file (bool): save file to local or not.
            run_name (str): name of rum for WandB.
        """
        # output config
        self.num_epochs = num_epochs
        self.use_focal = use_focal
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.save_file = save_file
        self.run_name = run_name

        self.annotations_path = '../data/annotations'
        self.images_path = '../data/images'
        self.focal_str = 'focal' if use_focal else 'no_focal'
        self.output_path = "results/{}epochs-{}/".format(self.num_epochs, self.focal_str)
        self.log_path     = self.output_path + "log"
        self.plot_output  = self.output_path + "scores.png"

        self.train_loss_output = self.output_path + "train_loss.npy"
        self.mAP_output = self.output_path + "mAP.npy"
        self.mAR_output = self.output_path + "mAR.npy"
 