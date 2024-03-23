"""
Here we create a simple 2D convolutional neural network. This network
is essentially going to try to estimate the diffusion process --- we
can then use this network to generate realistic images.

First, we create a single CNN block which we will stack to create the
full network. We use `LayerNorm` for stable training and no batch dependence.
"""

import torch
import torch.nn as nn
import numpy as np


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expected_shape: tuple[int, int],
        act: nn.Module = nn.GELU,
        kernel_size: int = 7,
    ):
        """
        A convolutional block for a larger CNN model.

        Parameters
        ----------
        in_channels : int
            The number of input channels. For grascale images, this is 1. For RGB images, this is 3.
        out_channels : int
            The number of output channels / feature maps produced by the convolutional layer. I.e., the
            number of kernels applied to the input tensor.
        expected_shape : tuple[int]
            The expected spatial dimensions of the last 2 dimensions (height, width) of the output tensor.
            This is used to create a `LayerNorm` layer to normalize the output of the convolutional layer.
        act : nn.Module
            The activation function to use.
        kernel_size : int
            The size of the kernel to be used in the convolutional layer. Default is 7x7 kernel.


        Examples
        --------
        >>> block = CNNBlock(in_channels=1, out_channels=16, expected_shape=(28, 28), act=nn.GELU, kernel_size=7)
        >>> input_tensor = torch.randn(128, 1, 28, 28)  # Example input batch of size (batch_size, channels, height, width)
        >>> output_tensor = block(input_tensor)
        """
        super().__init__()

        # comments below assumes batch_size=128 in_channels=1, out_channels=16, expected_shape=(28, 28), kernel_size=7
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size
                // 2,  # creates 3 cell padding around the image, 0 padding
            ),  # the Conv2d layer will output a tensor of shape (128, 16, 28, 28)
            # ^ (batch_size, out_channels, height, width)
            # LayerNorm normalizes the output of the convolutional layer, it will output a tensor of
            # shape (128, 16, 28, 28), where each batch has been normalised to standard normal
            nn.LayerNorm((out_channels, *expected_shape)),
            act(),
        )

    def forward(self, x):
        return self.net(x)


"""
We then create the full CNN model, which is a stack of these blocks
according to the `n_hidden` tuple, which specifies the number of
channels at each hidden layer.
"""


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expected_shape: tuple[int, int] = (28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        """
        A CNN designed to be used within a Denoising Diffusion Probabilistic Model (DDPM) as the
        generative network (decoder).

        This network incorporates a series of convolutional blocks (`CNNBlock`), followed by a final
        convolutional layer to produce the output. It also includes a mechanism for time embedding,
        allowing the network to adjust its behavior based on the current time step in the decoder process.

        Parameters
        ----------
        in_channels : int
            The number of input channels. For grascale images, this is 1. For RGB images, this is 3.
        expected_shape : tuple[int]
            The expected spatial dimensions of the last 2 dimensions (height, width) of the output tensor.
        n_hidden : tuple[int]
            The number of channels at each hidden layer. This is used to specify the number of output channels
            for each convolutional block in the network.
        kernel_size : int
            The size of the kernel to be used in the convolutional layer. Default is 7x7 kernel.
        last_kernel_size : int
            The size of the kernel to be used in the final convolutional layer. Default is 3x3 kernel.
        time_embeddings : int
            The number of time embeddings to use. This is used to create a time-dependent embedding for each
            time step in the diffusion process, allowing the network to adjust its processing based on the
            specific stage of the decoder process.
        act : nn.Module
            The activation function to use.

        Examples
        --------
        >>> cnn = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(64, 128, 64), kernel_size=7,
        >>>         last_kernel_size=3, time_embeddings=16, act=nn.GELU)
        >>> input_tensor = torch.randn(128, 1, 28, 28)  # Example input batch
        >>> time_steps = torch.randint(0, 1000, (128,))  # Example time steps for each image in the batch
        >>> output_tensor = cnn(input_tensor, time_steps)

        """
        super().__init__()
        last = in_channels  # The number of input channels for the first convolutional block

        # Create a series of convolutional blocks, each with the specified number of hidden channels
        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,  # No. of input channels for final Conv2d layer
                    hidden,  # No. of output channels for final Conv2d layer
                    expected_shape=expected_shape,  # default is (28, 28)
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = (
                hidden  # The number of input channels for the next convolutional block
            )

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,  # No. of input channels for final Conv2d layer
                in_channels,  # No. of output channels for final Conv2d layer, should be the same as input
                last_kernel_size,  # The size of the kernel to be used in the final convolutional layer
                padding=last_kernel_size // 2,
            )
        )

        ## This part is literally just to put the single scalar "t" into the CNN
        ## in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, n_hidden[0]),
        )

        # TODO: Unsure what the purpose of this is
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer('frequencies', frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generates a time-dependent embedding for each time step in the diffusion process.

        This function creates a sinusoidal embedding of the time step across multiple frequencies,
        enabling the CNN to adjust its processing based on the specific stage of the decoder
        process. The embedding is then transformed through several linear layers to produce
        a feature map that has the same shape as the first hidden layer.

        Parameters
        ----------
        t : torch.Tensor
            The time step for each item in the batch. This tensor should have shape (batch,).

        Returns
        -------
        torch.Tensor
            A time-dependent embedding for each time step in the diffusion process. This tensor
            has shape (batch, n_hidden[0], 1, 1).
        """
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN model.

        Passes the input tensor x through the first block of the CNN, then adds a time-dependent
        embedding for each time step in the decoder process. The tensor is then passed through
        the remaining blocks in the CNN to produce the output tensor.

        Parameters
        ----------
        x : torch.Tensor
            shape is (batch, chan, height, width).
        t : torch.Tensor
            shape is (batch,). Where each element represents the time step in the decoder process.
            for each item the bach.

        Returns
        -------
        torch.Tensor
            The output tensor from the CNN. This tensor has shape (batch, n_hidden[0], height, width).
            shape is (batch, chan, height, width).
        """

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed


class CNNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_hidden: tuple[int, ...],
        n_classes: int,
        expected_shape: tuple[int, int] = (28, 28),
        kernel_size: int = 5,
        adaptive_pooling_output_size: tuple[int, int] = (1, 1),
        act=nn.GELU,
    ):
        super().__init__()
        last = in_channels  # The number of input channels for the first convolutional block

        self.blocks = nn.ModuleList()

        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,  # No. of input channels for final Conv2d layer
                    hidden,  # No. of output channels for final Conv2d layer
                    expected_shape=expected_shape,  # default is (28, 28)
                    kernel_size=kernel_size,
                    act=act,
                )
            )

            last = (
                hidden  # The number of input channels for the next convolutional block
            )

        self.adaptive_pooling = nn.AdaptiveAvgPool2d(adaptive_pooling_output_size)

        # fully connected layer to output class probabilities
        self.final = nn.Linear(last * np.prod(adaptive_pooling_output_size), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        # apply adaptive pooling to get a fixed size tensor
        x = self.adaptive_pooling(x)

        # flatten the tensor for the final layer
        x = x.view(x.shape[0], -1)
        output = self.final(x)

        return output
