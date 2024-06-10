import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.dropout = nn.Dropout(dropout)
        # Delay the creation of conv to the forward method
        self.conv = None

    def forward(self, x):
        # Dynamically create the conv layer based on the input dimensions
        if self.conv is None:
            n_channels = x.shape[1]  # Dynamically get the number of input channels
            self.conv = nn.Conv1d(n_channels, self.d_model, kernel_size=self.patch_len,
                                  stride=self.stride, padding=self.padding).to(x.device)
        x = self.conv(x)  # [batch_size, d_model, new_seq_len]
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = (patch_len - stride) // 2

        # Patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Replacing conv_layers with fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(configs.d_model, configs.d_model)
            for _ in range(configs.e_layers)
        ])
        self.residual_layers = [nn.Identity() for _ in range(configs.e_layers)]

        # Placeholder for dynamically adjusting the linear layer
        self.num_features_after_fc = None  
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Patching and embedding
        x_enc = self.patch_embedding(x_enc.permute(0, 2, 1))

        # Adjusting for fully connected layers
        x_enc = x_enc.permute(0, 2, 1)  # Reshape for Linear: [batch_size, new_seq_len, d_model]
        x_enc = x_enc.reshape(x_enc.shape[0], x_enc.shape[1], -1)  # Flatten for Linear input

        for fc, residual in zip(self.fc_layers, self.residual_layers):
            residual_x = x_enc
            x_enc = fc(x_enc)
            x_enc += residual(residual_x)  # Apply residual connection

        # Flatten the output to prepare it for the linear layer
        output = self.flatten(x_enc)

        # Dynamically adjust the linear layer
        if self.linear is None or self.num_features_after_fc != output.shape[1]:
            self.num_features_after_fc = output.shape[1]
            # Adjust the linear layer based on the task
            self.linear = self.adjust_linear_layer(output)

        output = self.linear(output)  # Pass through the linear layer
        output = self.reshape_output_based_on_task(output)
        return output

    def adjust_linear_layer(self, output):
        # Logic to adjust linear layer based on task
        # Placeholder logic; please implement based on your actual requirements
        return nn.Linear(self.num_features_after_fc, self.pred_len * 20).to(output.device)

    def reshape_output_based_on_task(self, output):
        # Logic to reshape output based on task
        # Placeholder logic; please implement based on your actual requirements
        return output.view(-1, self.pred_len, 20)

