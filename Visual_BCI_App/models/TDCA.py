import numpy as np
from scipy import signal
from scipy.linalg import eigh
from scipy.linalg import qr
from scipy.linalg import solve
from scipy.stats import pearsonr


class _Signal:
	def __init__(self):
		self._callbacks = []

	def connect(self, callback):
		self._callbacks.append(callback)

	def emit(self, value):
		for callback in self._callbacks:
			callback(value)


def _is_pd(mat):
	try:
		_ = np.linalg.cholesky(mat)
		return True
	except np.linalg.LinAlgError:
		return False


def _nearest_pd(mat):
	b = (mat + mat.T) / 2.0
	_, s, v = np.linalg.svd(b)
	h = v.T @ np.diag(s) @ v
	a2 = (b + h) / 2.0
	a3 = (a2 + a2.T) / 2.0
	if _is_pd(a3):
		return a3

	spacing = np.spacing(np.linalg.norm(mat))
	eye = np.eye(mat.shape[0])
	k = 1
	while not _is_pd(a3):
		min_eig = np.min(np.real(np.linalg.eigvals(a3)))
		a3 += eye * (-min_eig * (k ** 2) + spacing)
		k += 1
	return a3


def _robust_pattern(w, cx, cs):
	return solve(cs.T, (cx @ w).T).T


def _xiang_dsp_kernel(x, y):
	x, y = np.asarray(x, dtype=float), np.asarray(y)
	labels = np.unique(y)
	x = np.reshape(x, (-1, *x.shape[-2:]))
	np.subtract(x, np.mean(x, axis=-1, keepdims=True), out=x)

	n_labels = np.array([np.sum(y == label) for label in labels])
	m = np.mean(x, axis=0)
	ms, ss = zip(
		*[
			(
				np.mean(x[y == label], axis=0),
				np.sum(
					np.matmul(x[y == label], np.swapaxes(x[y == label], -1, -2)),
					axis=0,
				),
			)
			for label in labels
		]
	)
	ms, ss = np.stack(ms), np.stack(ss)

	sw = np.sum(
		ss - n_labels[:, np.newaxis, np.newaxis] * np.matmul(ms, np.swapaxes(ms, -1, -2)),
		axis=0,
	)
	ms = ms - m
	sb = np.sum(
		n_labels[:, np.newaxis, np.newaxis] * np.matmul(ms, np.swapaxes(ms, -1, -2)),
		axis=0,
	)

	d, w = eigh(_nearest_pd(sb), _nearest_pd(sw))
	ix = np.argsort(d)[::-1]
	d, w = d[ix], w[:, ix]
	_ = _robust_pattern(w, sb, w.T @ sb @ w)
	return w, d, m


def _xiang_dsp_feature(w, m, x, n_components):
	w, m, x = np.asarray(w, dtype=float), np.asarray(m, dtype=float), np.asarray(x, dtype=float)
	x = np.reshape(x, (-1, *x.shape[-2:]))
	np.subtract(x, np.mean(x, axis=-1, keepdims=True), out=x)
	return np.matmul(w[:, :n_components].T, x - m)


class TDCA:
	"""
	Official supervised TDCA core.
	Inference requires fit(X, y) beforehand.
	"""

	def __init__(self, num_harmonics, times, targets, Nh=8, lagging_len=None, sample_rate=250, delay_sec=0.14, n_components=3):
		self.sendResultSignal = _Signal()

		self.Nh = int(Nh)
		self.Fs = int(sample_rate)
		self.targets = [float(x) for x in targets]
		self.Nf = len(self.targets)
		self.classes_ = np.arange(self.Nf)

		self.delay_sec = float(delay_sec)
		self.ws = max(float(times) - self.delay_sec, 0.2)
		self.T = int(round(self.Fs * self.ws))

		self.Nm = 8 if self.Fs <= 300 else 10
		self.n_components = int(n_components)
		full_points = int(round(float(times) * self.Fs))
		if lagging_len is not None:
			self.lagging_len = int(lagging_len)
		else:
			self.lagging_len = min(8, max(1, full_points - self.T))
		self.required_points = int(self.T + self.lagging_len)

		self.reference_signals = self.get_reference_signal(num_harmonics, self.targets)
		self.Ps = [self._proj_ref(self.reference_signals[i]) for i in range(self.Nf)]
		self.frequency_weights = np.ones(self.Nf, dtype=float)

		self._sos_filters = self._design_filter_bank()

		self._is_fitted = False
		self.W = []
		self.M = []
		self.templates = []

	def get_reference_signal(self, num_harmonics, targets):
		reference_signals = []
		t = np.arange(0, (self.T / self.Fs), step=1.0 / self.Fs)
		for f in targets:
			ref_f = []
			for h in range(1, int(num_harmonics) + 1):
				ref_f.append(np.sin(2 * np.pi * h * f * t)[0:self.T])
				ref_f.append(np.cos(2 * np.pi * h * f * t)[0:self.T])
			reference_signals.append(ref_f)
		return np.asarray(reference_signals, dtype=float)

	def _proj_ref(self, yf):
		q, _ = qr(yf.T, mode="economic")
		return q @ q.T

	def _design_filter_bank(self):
		nyq = self.Fs / 2.0
		pass_band = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
		stop_band = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
		high_cut_pass = min(80, int(self.Fs * 0.45))
		high_cut_stop = min(90, int(self.Fs * 0.48))
		gpass, gstop, rp = 3, 40, 0.5

		filters = []
		for i in range(self.Nm):
			wp = np.array([pass_band[i] / nyq, high_cut_pass / nyq], dtype=float)
			ws = np.array([stop_band[i] / nyq, high_cut_stop / nyq], dtype=float)
			n, wn = signal.cheb1ord(wp, ws, gpass, gstop)
			b, a = signal.cheby1(n, rp, wn, "bandpass")
			filters.append((b, a))
		return filters

	def _lagging_aug(self, x, n_samples, lagging_len, p, training):
		x = x.reshape((-1, *x.shape[-2:]))
		n_trials, n_channels, n_points = x.shape
		if n_points < n_samples:
			raise ValueError("the length of X should be larger than n_samples.")

		# 当输入窗口短于 n_samples+lag 时，自动降低可用 lag，避免固定lag导致训练/推理直接失败。
		effective_lag = min(int(lagging_len), max(0, int(n_points - n_samples)))

		aug_x = np.zeros((n_trials, (effective_lag + 1) * n_channels, n_samples), dtype=float)
		if training:
			for i in range(effective_lag + 1):
				aug_x[:, i * n_channels:(i + 1) * n_channels, :] = x[..., i:i + n_samples]
		else:
			for i in range(effective_lag + 1):
				aug_x[:, i * n_channels:(i + 1) * n_channels, :n_samples - i] = x[..., i:n_samples]

		aug_xp = aug_x @ p
		return np.concatenate([aug_x, aug_xp], axis=-1)

	def _filter_bank_batch(self, x):
		x = np.asarray(x, dtype=float)
		x = np.reshape(x, (-1, *x.shape[-2:]))
		x = x[:, :, -self.required_points:]
		fb = np.zeros((self.Nm, x.shape[0], x.shape[1], x.shape[2]), dtype=float)
		for i, (b, a) in enumerate(self._sos_filters):
			fb[i] = signal.filtfilt(b, a, x, axis=-1, padlen=3 * (max(len(b), len(a)) - 1)).copy()
		return fb

	def filter_bank(self, eeg):
		return self._filter_bank_batch(np.asarray(eeg, dtype=float)[np.newaxis, ...])[:, 0]

	def fit(self, x, y):
		x = np.asarray(x, dtype=float)
		y = np.asarray(y).reshape(-1)
		x = np.reshape(x, (-1, *x.shape[-2:]))
		if x.shape[0] != y.shape[0]:
			raise ValueError("X and y length mismatch")

		self.W, self.M, self.templates = [], [], []
		fb_train = self._filter_bank_batch(x)

		for fb_i in range(self.Nm):
			x_fb = fb_train[fb_i] - np.mean(fb_train[fb_i], axis=-1, keepdims=True)
			aug_x_list, aug_y_list = [], []
			for i, label in enumerate(self.classes_):
				class_x = x_fb[y == label]
				if class_x.shape[0] == 0:
					raise ValueError(f"missing class samples for label {int(label)}")
				aug_x_list.append(self._lagging_aug(class_x, self.Ps[i].shape[0], self.lagging_len, self.Ps[i], training=True))
				aug_y_list.append(np.full(class_x.shape[0], label, dtype=int))

			aug_x = np.concatenate(aug_x_list, axis=0)
			aug_y = np.concatenate(aug_y_list, axis=0)

			w_fbi, _, m_fbi = _xiang_dsp_kernel(aug_x, aug_y)
			self.W.append(w_fbi)
			self.M.append(m_fbi)

			template_i = []
			for label in self.classes_:
				feat = _xiang_dsp_feature(w_fbi, m_fbi, aug_x[aug_y == label], n_components=min(self.n_components, w_fbi.shape[1]))
				template_i.append(np.mean(feat, axis=0))
			self.templates.append(np.stack(template_i, axis=0))

		self._is_fitted = True
		return self

	def _tdca_feature(self, x, templates, w, m, ps, lagging_len, n_components, training=False):
		rhos = []
		for xk, p in zip(templates, ps):
			a = _xiang_dsp_feature(
				w,
				m,
				self._lagging_aug(x, p.shape[0], lagging_len, p, training=training),
				n_components=n_components,
			)
			b = xk[:n_components, :]
			a = np.reshape(a, (-1,))
			b = np.reshape(b, (-1,))
			try:
				rho = pearsonr(a, b)[0]
			except Exception:
				rho = 0.0
			if not np.isfinite(rho):
				rho = 0.0
			rhos.append(float(rho))
		return np.asarray(rhos, dtype=float)

	def _transform(self, x, fb_i):
		x = np.asarray(x, dtype=float)
		x = np.reshape(x, (-1, *x.shape[-2:]))
		x = x - np.mean(x, axis=-1, keepdims=True)

		n_comp = min(self.n_components, self.W[fb_i].shape[1])
		rhos = [
			self._tdca_feature(
				trial,
				self.templates[fb_i],
				self.W[fb_i],
				self.M[fb_i],
				self.Ps,
				self.lagging_len,
				n_components=n_comp,
				training=False,
			)
			for trial in x
		]
		return np.stack(rhos, axis=0)

	def _score_vector_fitted(self, test_data):
		x = np.asarray(test_data, dtype=float)
		x = np.reshape(x, (-1, *x.shape[-2:]))
		fb_test = self._filter_bank_batch(x)

		if self.Nm == 1:
			sum_features = self._transform(fb_test[0], 0)
		else:
			sum_features = np.zeros((self.Nm, x.shape[0], self.Nf), dtype=float)
			for fb_i in range(self.Nm):
				fb_weight = (fb_i + 1) ** (-1.25) + 0.25
				sum_features[fb_i] = fb_weight * self._transform(fb_test[fb_i], fb_i)
			sum_features = np.sum(sum_features, axis=0)

		return sum_features[0]

	def _ensure_fitted(self):
		if not self._is_fitted:
			raise RuntimeError("TDCA model is not fitted. Please run fit(X, y) first.")

	def _confidence_from_scores(self, scores):
		scores = np.asarray(scores, dtype=float).reshape(-1)
		if scores.size < 2:
			return float(scores[0]) if scores.size == 1 else 0.0
		top2 = np.partition(scores, -2)[-2:]
		return float(top2[1] - top2[0])

	def score_vector(self, test_data):
		self._ensure_fitted()
		scores = self._score_vector_fitted(test_data)
		return np.asarray(scores, dtype=float) * self.frequency_weights

	def classify_with_scores(self, test_data):
		scores = self.score_vector(test_data)
		result = int(np.argmax(scores))
		confidence = self._confidence_from_scores(scores)
		return result, scores, confidence

	def classify_4_class_with_scores(self, test_data):
		full_scores = self.score_vector(test_data)
		scores = np.asarray(full_scores[:4], dtype=float)
		result = int(np.argmax(scores))
		confidence = self._confidence_from_scores(scores)
		return result, scores, confidence

	def classify(self, test_data):
		result, _, _ = self.classify_with_scores(test_data)
		return int(result)

	def classify_4_class(self, test_data):
		result, _, _ = self.classify_4_class_with_scores(test_data)
		return int(result)

	def set_frequency_weight(self, freq_index, weight):
		if 0 <= freq_index < self.Nf:
			self.frequency_weights[freq_index] = float(weight)

	def set_frequency_weights(self, weights):
		arr = np.asarray(weights, dtype=float).reshape(-1)
		if arr.shape[0] != self.Nf:
			raise ValueError(f"weights length mismatch: expected {self.Nf}, got {arr.shape[0]}")
		self.frequency_weights = arr.copy()

	def get_frequency_weights(self):
		return self.frequency_weights.copy()

	def reset_frequency_weights(self):
		self.frequency_weights = np.ones(self.Nf, dtype=float)

	def clear_fit(self):
		self._is_fitted = False
		self.W, self.M, self.templates = [], [], []

	@property
	def is_fitted(self):
		return bool(self._is_fitted)


# ============================================================================
# Shrinkage utilities
# ============================================================================

def _ledoit_wolf_shrinkage(X):
	"""Ledoit-Wolf shrinkage estimator for covariance matrix.

	X: (n_features, n_samples) — centered data.
	Returns regularized covariance estimate.
	"""
	n_features, n_samples = X.shape
	S = (X @ X.T) / n_samples  # sample covariance
	m = np.trace(S) / n_features
	d2 = np.sum((S - m * np.eye(n_features)) ** 2)
	b2 = np.sum([np.sum((np.outer(X[:, i], X[:, i]) - S) ** 2)
	              for i in range(n_samples)]) / (n_samples ** 2)
	shrinkage = max(0.0, min(1.0, b2 / max(d2, 1e-10)))
	return (1 - shrinkage) * S + shrinkage * m * np.eye(n_features)


# ============================================================================
# Variant 1: Shrinkage-TDCA (Ledoit-Wolf regularization)
# ============================================================================

class TDCA_SHRINK(TDCA):
	"""TDCA with Ledoit-Wolf shrinkage instead of nearest-PD regularization.

	Shrinkage provides better small-sample stability, especially for
	short data windows where the number of training trials is limited.
	"""

	def fit(self, x, y):
		x = np.asarray(x, dtype=float)
		y = np.asarray(y).reshape(-1)
		x = np.reshape(x, (-1, *x.shape[-2:]))
		if x.shape[0] != y.shape[0]:
			raise ValueError("X and y length mismatch")

		self.W, self.M, self.templates = [], [], []
		fb_train = self._filter_bank_batch(x)

		for fb_i in range(self.Nm):
			x_fb = fb_train[fb_i] - np.mean(fb_train[fb_i], axis=-1, keepdims=True)
			aug_x_list, aug_y_list = [], []
			for i, label in enumerate(self.classes_):
				class_x = x_fb[y == label]
				if class_x.shape[0] == 0:
					raise ValueError(f"missing class samples for label {int(label)}")
				aug_x_list.append(self._lagging_aug(class_x, self.Ps[i].shape[0],
				                                    self.lagging_len, self.Ps[i], training=True))
				aug_y_list.append(np.full(class_x.shape[0], label, dtype=int))

			aug_x = np.concatenate(aug_x_list, axis=0)
			aug_y = np.concatenate(aug_y_list, axis=0)

			# Shrinkage-based DSP kernel: use LW shrinkage for scatter matrices
			w_fbi, _, m_fbi = _xiang_dsp_kernel_shrinkage(aug_x, aug_y)
			self.W.append(w_fbi)
			self.M.append(m_fbi)

			template_i = []
			n_comp = min(self.n_components, w_fbi.shape[1])
			for label in self.classes_:
				feat = _xiang_dsp_feature(w_fbi, m_fbi,
				                          aug_x[aug_y == label],
				                          n_components=n_comp)
				template_i.append(np.mean(feat, axis=0))
			self.templates.append(np.stack(template_i, axis=0))

		self._is_fitted = True
		return self


def _xiang_dsp_kernel_shrinkage(x, y):
	"""DSP kernel variant using shrinkage regularization on scatter matrices."""
	x, y = np.asarray(x, dtype=float), np.asarray(y)
	labels = np.unique(y)
	x = np.reshape(x, (-1, *x.shape[-2:]))
	np.subtract(x, np.mean(x, axis=-1, keepdims=True), out=x)

	n_labels = np.array([np.sum(y == label) for label in labels])
	m = np.mean(x, axis=0)
	ms, ss = zip(*[(np.mean(x[y == label], axis=0),
	                np.sum(np.matmul(x[y == label], np.swapaxes(x[y == label], -1, -2)), axis=0))
	               for label in labels])
	ms, ss = np.stack(ms), np.stack(ss)

	sw = np.sum(ss - n_labels[:, np.newaxis, np.newaxis] *
	            np.matmul(ms, np.swapaxes(ms, -1, -2)), axis=0)
	ms_centered = ms - m
	sb = np.sum(n_labels[:, np.newaxis, np.newaxis] *
	            np.matmul(ms_centered, np.swapaxes(ms_centered, -1, -2)), axis=0)

	d = sw.shape[0]
	# Apply ridge-style shrinkage: Sw_reg = (1-alpha)*Sw + alpha*trace(Sw)/d * I
	alpha = 0.1
	tr_sw = np.trace(sw) / d
	sw_reg = (1 - alpha) * sw + alpha * tr_sw * np.eye(d)

	try:
		d_vals, w = eigh(_nearest_pd(sb), sw_reg)
	except np.linalg.LinAlgError:
		d_vals, w = eigh(_nearest_pd(sb), _nearest_pd(sw))

	ix = np.argsort(d_vals)[::-1]
	d_vals, w = d_vals[ix], w[:, ix]
	return w, d_vals, m


# ============================================================================
# Variant 2: Multi-Lag Ensemble TDCA
# ============================================================================

class MultiLagTDCA:
	"""Ensemble of TDCA models with different lag values.

	Trains multiple TDCA models {lag=4, 8, 12} and combines their
	score vectors via weighted averaging.
	"""

	def __init__(self, num_harmonics, times, targets, Nh=8,
	             lags=None, sample_rate=250, delay_sec=0.14,
	             n_components=3, lagging_len=None):
		self.lags = lags if lags is not None else [4, 8, 12]
		self.models = []

		for lag in self.lags:
			model = TDCA(num_harmonics, times, targets, Nh=Nh,
			             lagging_len=lag, sample_rate=sample_rate,
			             delay_sec=delay_sec, n_components=n_components)
			self.models.append(model)

		self.targets = [float(x) for x in targets]
		self.Nf = len(self.targets)
		self.Nm = self.models[0].Nm if self.models else 8
		self.frequency_weights = np.ones(self.Nf, dtype=float)
		# Expose lagging_len/required_points for compatibility
		# Use max lag to ensure enough data for all ensemble members
		self.lagging_len = max(self.lags) if self.lags else 8
		base_model = self.models[-1] if self.models else None  # model with largest lag
		self.required_points = base_model.required_points if base_model else 100

	@property
	def is_fitted(self):
		return all(m.is_fitted for m in self.models)

	def fit(self, x, y):
		for model in self.models:
			model.fit(x, y)
		return self

	def score_vector(self, test_data):
		# Average score vectors from all lag models
		all_scores = []
		for model in self.models:
			all_scores.append(model.score_vector(test_data))
		return np.mean(all_scores, axis=0) * self.frequency_weights

	def classify(self, test_data):
		scores = self.score_vector(test_data)
		return int(np.argmax(scores))

	def classify_with_scores(self, test_data):
		scores = self.score_vector(test_data)
		result = int(np.argmax(scores))
		if scores.size >= 2:
			top2 = np.partition(scores, -2)[-2:]
			confidence = float(top2[1] - top2[0])
		else:
			confidence = float(scores[0]) if scores.size == 1 else 0.0
		return result, scores, confidence

	def clear_fit(self):
		for model in self.models:
			model.clear_fit()

	def set_frequency_weights(self, weights):
		arr = np.asarray(weights, dtype=float).reshape(-1)
		if arr.shape[0] != self.Nf:
			raise ValueError(f"weights length mismatch")
		self.frequency_weights = arr.copy()
