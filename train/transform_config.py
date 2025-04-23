import torchvision.transforms as TF


ni_img_size = (64,64) # natural image
mi_img_size = (224,224) # medical image
mnist_img_size = (32,32)    # mnist


cifar_transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize(mnist_img_size, 
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True),
            TF.Lambda(lambda t: (t * 2) - 1) 
        ]
    )

celeba_transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize(ni_img_size, 
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True),
            TF.Lambda(lambda t: (t * 2) - 1)
        ]
    )



MNIST_transforms_mnistvar = TF.Compose(
        [
            TF.Grayscale(num_output_channels=1),
            TF.ToTensor(),
            TF.Resize(mnist_img_size, 
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True),
            # TF.RandomHorizontalFlip(),
            TF.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
        ]
    )
    

    
CHE_transforms_che = TF.Compose(
        [
            TF.Grayscale(num_output_channels=1),
            TF.ToTensor(),
            TF.Resize(mi_img_size,
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True
                ),
            TF.Lambda(lambda t: (t * 2) - 1)
        ]
    )
