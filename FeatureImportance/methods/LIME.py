import lime.lime_tabular
import numpy as np
import torch

class LIME:
    def __init__(self, model, output_dim):
        self.model = model
        self.output_dim = output_dim
        self.explainers = [None for _ in range(output_dim)]

    def _f(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            return self.model(X).cpu().numpy()

    def _get_output_fn(self, output_idx):
        def output_fn(X):
            return self._f(X)[:, output_idx]
        return output_fn

    def fit_lime_exp(self, background, mode="regression"):
        for i in range(self.output_dim):
            self.explainers[i] = lime.lime_tabular.LimeTabularExplainer(
                training_data=background,
                mode=mode,
                discretize_continuous=False
            )

    def get_single_FI(self, point, output_idx):
        if self.explainers[output_idx] is None:
            raise RuntimeError(f"LIME explainer not initialized for output index {output_idx}.")

        predict_fn = self._get_output_fn(output_idx)

        explanation = self.explainers[output_idx].explain_instance(
            data_row=point,
            predict_fn=predict_fn,
            num_features=len(point),
            num_samples=1000
        )

        fi_map = dict(explanation.as_map()[0])  # for regression
        fi_vector = np.array([fi_map.get(i, 0.0) for i in range(len(point))])

        pred = predict_fn(point.reshape(1, -1))[0]
        return pred, fi_vector
    
    def get_fi(self, X):
        """
        Returns:
        - preds: [n_points, output_dim]
        - fis: [n_points, output_dim, num_features]
        """
        if None in self.explainers:
            self.fit_lime_exp(X)
        n_points = len(X)
        num_features = X.shape[1]
        preds = np.zeros((n_points, self.output_dim))
        fis = np.zeros((n_points, self.output_dim, num_features))

        for i, point in enumerate(X):
            for output_idx in range(self.output_dim):
                pred, fi = self.get_single_FI(point, output_idx)
                preds[i, output_idx] = pred
                fis[i, output_idx, :] = fi

        return preds, np.stack(fis)