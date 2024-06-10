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

        # LSTM Encoder
        self.lstm = nn.LSTM(input_size=configs.d_model,
                            hidden_size=configs.d_model,
                            num_layers=configs.e_layers,
                            batch_first=True,
                            dropout=configs.dropout if configs.e_layers > 1 else 0)

        # Dynamically calculate the number of features to be fed into the linear layer
        self.num_features_after_lstm = None  # This will be set dynamically

        # Prediction Head - The linear layer will be defined later, in the forward method
        self.flatten = nn.Flatten(start_dim=-2)
        # Placeholder for the linear layer
        self.linear = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Perform patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # Prepare for Conv1d
        enc_out = self.patch_embedding(x_enc)  # Apply patch embedding

        # LSTM Encoder processing
        enc_out = enc_out.permute(0, 2, 1)  # Prepare for LSTM: [batch_size, seq_len, feature_dim]
        enc_out, (hidden, cell) = self.lstm(enc_out)

        # Flatten the output of LSTM to prepare it for the linear layer
        output = self.flatten(enc_out)

        # Dynamically create or adjust the linear layer based on the actual number of features
        if self.linear is None or self.num_features_after_lstm != output.shape[1]:
            self.num_features_after_lstm = output.shape[1]
            if self.task_name in ['long_term_forecast', 'short_term_forecast']:
                self.linear = nn.Linear(self.num_features_after_lstm, self.pred_len*20).to(output.device)
            elif self.task_name in ['imputation', 'anomaly_detection']:
                self.linear = nn.Linear(self.num_features_after_lstm, self.seq_len).to(output.device)
            elif self.task_name == 'classification':
                self.linear = nn.Linear(self.num_features_after_lstm, configs.num_class).to(output.device)

        output = self.linear(output)  # Pass through the linear layer
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection']:
            output = output.view(-1, self.pred_len, 20)
        # elif self.task_name == 'classification':
        #     # For classification, L could be 1, and D the number of classes, assuming one prediction per sequence
        #     output = output.view(-1, configs.num_class)
        # Reshape or process output if necessary for specific tasks
        # (You may need to adjust the reshaping based on your specific task requirements)
        return output

