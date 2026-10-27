"""Lazy PyTorch autoencoder transformer; serialized state uses NumPy arrays."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Autoencoder(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=2, hidden_size=32, max_epochs=20, batch_size=32,
                 learning_rate=.001, random_state=42):
        self.n_components = n_components
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _module(self, torch, width):
        return torch.nn.Sequential(torch.nn.Linear(width, self.hidden_size), torch.nn.ReLU(),
                                   torch.nn.Linear(self.hidden_size, self.n_components),
                                   torch.nn.Linear(self.n_components, self.hidden_size), torch.nn.ReLU(),
                                   torch.nn.Linear(self.hidden_size, width))

    def fit(self, X, y=None):
        from .sequence_models import deep_cpu_context
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Autoencoder needs a finite 2D numeric matrix")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (self.n_components, self.hidden_size, self.max_epochs, self.batch_size)) or self.max_epochs > 100:
            raise ValueError("Autoencoder dimensions/batch/epochs must be positive integers; epochs <= 100")
        if self.n_components > values.shape[1]:
            raise ValueError("Autoencoder n_components must not exceed input width")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        with deep_cpu_context(self.random_state) as torch:
            module = self._module(torch, values.shape[1])
            optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)
            tensor = torch.as_tensor(values)
            self.loss_curve_ = []
            module.train()
            for _ in range(self.max_epochs):
                permutation = torch.randperm(len(values))
                total = 0.
                for indices in permutation.split(self.batch_size):
                    batch = tensor[indices]
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(module(batch), batch)
                    if not torch.isfinite(loss):
                        raise ValueError("Autoencoder training loss is not finite")
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(batch)
                self.loss_curve_.append(total / len(values))
            self.state_ = {name: value.detach().cpu().numpy().copy() for name, value in module.state_dict().items()}
        self.n_features_in_ = values.shape[1]
        return self

    def _apply(self, X, reconstruct=False):
        from .sequence_models import deep_cpu_context
        check_is_fitted(self, "state_")
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError("Autoencoder input does not match its finite fitted feature matrix")
        with deep_cpu_context() as torch:
            module = self._module(torch, self.n_features_in_)
            module.load_state_dict({name: torch.as_tensor(value) for name, value in self.state_.items()})
            module.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(values)
                result = module(tensor) if reconstruct else module[:3](tensor)
                return result.detach().cpu().numpy()

    def transform(self, X):
        return self._apply(X)

    def reconstruct(self, X):
        return self._apply(X, reconstruct=True)
