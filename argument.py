import argparse
import os


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        "Train IQA GAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for generator")

    parser.add_argument("--generator_optim_file", action="store", type=str,
                        default=None,
                        help="saved state for generator optimizer")

    parser.add_argument("--shadow_generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for the shadow generator")

    parser.add_argument("--discriminator_file", action="store", type=str,
                        default=None,
                        help="pretrained_weights file for discriminator")

    parser.add_argument("--discriminator_optim_file", action="store", type=str,
                        default=None,
                        help="saved state for discriminator optimizer")

    parser.add_argument("--images_dir", action="store", type=str,
                        default="./data/images",
                        # default=os.environ['SM_CHANNEL_TRAINING'],
                        help="path for the images directory")

    parser.add_argument("--folder_distributed", action="store", type=bool,
                        default=False,
                        help="whether the images directory contains folders or not")

    parser.add_argument("--flip_augment", action="store", type=bool,
                        default=True,
                        help="whether to randomly mirror the images during training")

    parser.add_argument("--sample_dir", action="store", type=str,
                        default="./data/samples/1/",
                        # default=os.environ['SM_MODEL_DIR'],
                        help="path for the generated samples directory")

    parser.add_argument("--model_dir", action="store", type=str,
                        default="./data/models/1/",
                        # default=os.environ['SM_MODEL_DIR'],
                        help="path for saved models directory")

    parser.add_argument("--loss_function", action="store", type=str,
                        default="relativistic-hinge",
                        help="loss function to be used: " +
                             "standard-gan, wgan-gp, lsgan," +
                             "lsgan-sigmoid," +
                             "hinge, relativistic-hinge")

    parser.add_argument("--depth", action="store", type=int,
                        default=7,
                        help="Depth of the GAN")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=512,
                        help="latent size for the generator")

    parser.add_argument("--batch_size", action="store", type=int,
                        default=20,
                        help="batch_size for training")

    parser.add_argument("--start", action="store", type=int,
                        default=1,
                        help="starting epoch number")

    parser.add_argument("--num_epochs", action="store", type=int,
                        default=3,
                        help="number of epochs for training")

    parser.add_argument("--feedback_factor", action="store", type=int,
                        default=100,
                        help="number of logs to generate per epoch")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=36,
                        help="number of samples to generate for creating the grid" +
                             " should be a square number preferably")

    parser.add_argument("--checkpoint_factor", action="store", type=int,
                        default=1,
                        help="save model per n epochs")

    parser.add_argument("--g_lr", action="store", type=float,
                        default=0.003,
                        help="learning rate for generator")

    parser.add_argument("--d_lr", action="store", type=float,
                        default=0.003,
                        help="learning rate for discriminator")

    parser.add_argument("--adam_beta1", action="store", type=float,
                        default=0,
                        help="value of beta_1 for adam optimizer")

    parser.add_argument("--adam_beta2", action="store", type=float,
                        default=0.99,
                        help="value of beta_2 for adam optimizer")

    parser.add_argument("--use_eql", action="store", type=bool,
                        default=True,
                        help="Whether to use equalized learning rate or not")

    parser.add_argument("--use_ema", action="store", type=bool,
                        default=True,
                        help="Whether to use exponential moving averages or not")

    parser.add_argument("--ema_decay", action="store", type=float,
                        default=0.999,
                        help="decay value for the ema")

    parser.add_argument("--data_percentage", action="store", type=float,
                        default=100,
                        help="percentage of data to use")

    parser.add_argument("--num_workers", action="store", type=int,
                        default=3,
                        help="number of parallel workers for reading files")

    args = parser.parse_args()

    return args
