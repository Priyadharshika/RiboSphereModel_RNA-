import torch.nn as nn


class FeedForward(nn.Module):
    """
    Two-layer feed-forward network used in the transformer.
    """

    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        activation=nn.GELU,
        dropout=0.0,
    ):
        super().__init__()

        if min(
            input_features,
            hidden_features,
            output_features
        ) <= 0:
            raise ValueError(
                "All feature dimensions must be positive."
            )

        self.fc1 = nn.Linear(
            input_features,
            hidden_features
        )

        self.activation = activation()

        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(
            hidden_features,
            output_features
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):

        outputs = self.fc1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout1(outputs)

        outputs = self.fc2(outputs)
        outputs = self.dropout2(outputs)

        return outputs