import os
import torch

from MSG_GAN.GAN import MSG_GAN
import MSG_GAN.Losses as Losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def solver(args, data):
    # create a gan from these
    msg_gan = MSG_GAN(depth=args.depth,
                      latent_size=args.latent_size,
                      use_eql=args.use_eql,
                      use_ema=args.use_ema,
                      ema_decay=args.ema_decay,
                      device=device)

    if args.generator_file is not None:
        # load the weights into generator
        print("loading generator_weights from:", args.generator_file)
        msg_gan.gen.load_state_dict(torch.load(args.generator_file))

    print("Generator Configuration: ")
    print(msg_gan.gen)

    if args.shadow_generator_file is not None:
        # load the weights into generator
        print("loading shadow_generator_weights from:",
              args.shadow_generator_file)
        msg_gan.gen_shadow.load_state_dict(
            torch.load(args.shadow_generator_file))

    if args.discriminator_file is not None:
        # load the weights into discriminator
        print("loading discriminator_weights from:", args.discriminator_file)
        msg_gan.dis.load_state_dict(torch.load(args.discriminator_file))

    print("Discriminator Configuration: ")
    print(msg_gan.dis)

    # create optimizer for generator:
    gen_optim = torch.optim.Adam(msg_gan.gen.parameters(), args.g_lr,
                                 (args.adam_beta1, args.adam_beta2))
    dis_optim = torch.optim.Adam(msg_gan.dis.parameters(), args.d_lr,
                                 (args.adam_beta1, args.adam_beta2))

    if args.generator_optim_file is not None:
        print("loading gen_optim_state from:", args.generator_optim_file)
        gen_optim.load_state_dict(torch.load(args.generator_optim_file))

    if args.discriminator_optim_file is not None:
        print("loading dis_optim_state from:", args.discriminator_optim_file)
        dis_optim.load_state_dict(torch.load(args.discriminator_optim_file))

    loss_name = args.loss_function.lower()

    if loss_name == "hinge":
        loss = Losses.HingeGAN
    elif loss_name == "relativistic-hinge":
        loss = Losses.RelativisticAverageHingeGAN
    elif loss_name == "standard-gan":
        loss = Losses.StandardGAN
    elif loss_name == "lsgan":
        loss = Losses.LSGAN
    elif loss_name == "lsgan-sigmoid":
        loss = Losses.LSGAN_SIGMOID
    elif loss_name == "wgan-gp":
        loss = Losses.WGAN_GP
    else:
        raise Exception("Unknown loss function requested")

    # train the GAN
    msg_gan.train(
        data,
        gen_optim,
        dis_optim,
        loss_fn=loss(msg_gan.dis),
        num_epochs=args.num_epochs,
        checkpoint_factor=args.checkpoint_factor,
        data_percentage=args.data_percentage,
        feedback_factor=args.feedback_factor,
        num_samples=args.num_samples,
        sample_dir=args.sample_dir,
        save_dir=args.model_dir,
        log_dir=args.model_dir,
        start=args.start
    )

