import torch


class CRF:
    def __init__(self, feature_size, num_labels):
        self.feature_size = feature_size
        self.num_labels = num_labels
        self._linear = torch.nn.Linear(feature_size, num_labels)
        self._transitions = torch.randn(num_labels, num_labels)

    def forward(self, x, labels, mask):
        """
        x: (batch_size, seq_len, feature_size)
        labels: (batch_size, seq_len)
        mask: (batch_size, seq_len)
        """
        logits = self._linear(x)
        B, L, C = logits.shape
        mask = mask[:, :, None]
        log_alphas = torch.zeros_like(logits)
        log_betas = torch.zeros_like(logits)

        transitions = self._transitions.unsqueeze(0)

        log_alphas[:, 0, :] = logits[:, 0, :]
        for t in range(1, L):
            palpha = log_alphas[:, t - 1][:, :, None] + transitions + logits[:, t][:, None, :]
            palpha = torch.logsumexp(palpha, dim=1)
            log_alphas[:, t, :] = palpha * mask[:, t] + log_alphas[:, t - 1, :] * (1 - mask[:, t])

        log_betas[:, L - 1, :] = 0
        for t in range(L - 2, -1, -1):
            pbeta = log_betas[:, t + 1][:, None, :] + transitions + logits[:, t + 1][:, None, :]
            pbeta = torch.logsumexp(pbeta, dim=2)
            log_betas[:, t, :] = pbeta * mask[:, t + 1] + log_betas[:, t + 1, :] * (1 - mask[:, t + 1])

        potentials = log_alphas + log_betas
        input_likelihood = torch.logsumexp(log_alphas[:, -1], -1)
        log_marginals = potentials - input_likelihood[:, None, None]

        marginal_probs = torch.exp(log_marginals) #(B, L, C)

        binary_potentials = (
            log_alphas[:, :-1][:, :, :, None]
            + log_betas[:, 1:][:, :, None, :]
            + transitions.unsqueeze(0)
            + logits[:, 1:][:, :, None, :]
        ) #(B, L, C, C)

        log_binary_marginals = binary_potentials - input_likelihood[:, None, None, None]
        binary_probs = torch.exp(log_binary_marginals) #(B, L, C, C)

        weight_gradient = torch.zeros_like(self._linear.weight) #(F, C)
        bias_gradient = torch.zeros_like(self._linear.bias) #(C, )
        transition_gradient = torch.zeros_like(self._transitions) #(C, C)

        for t in range(L):
            weight_gradient += x[:, t, :, None] * (labels[:, t, None, :] - marginal_probs[:, t, None, :])

