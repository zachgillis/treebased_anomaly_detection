from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import plotting_utils as pf
import yaml


models = keras.models
layers = keras.layers
regularizers = keras.regularizers

def make_model(activation="relu", hidden=3, inputs=4, lr=1e-3, dropout=0, l1=0, l2=0, momentum=0.9, label_smoothing=0):
	model = models.Sequential()
	model.add(layers.Dense(64, input_shape=(inputs,)))
	for i in range(hidden-1):
		if activation =="relu":
			model.add(layers.ReLU())
		elif activation == "leaky":
			model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Dropout(dropout))
		model.add(layers.Dense(64,kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
	model.add(layers.Dense(2, activation="softmax"))

	loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

	model.compile(
		loss=loss,
		optimizer=keras.optimizers.Adam(lr, beta_1=momentum),
		metrics=["accuracy"],
		weighted_metrics=[],
	)
	return model


def _make_pretrain_model(params, args):
	return make_model(
		activation=params['activation'], hidden=int(params['hidden']), inputs=args.inputs,
		lr=float(params['lr']), dropout=float(params['dropout']), l1=float(params['l1']),
		l2=float(params['l2']), momentum=float(params['beta_1']),
		label_smoothing=float(params['label_smoothing']))


def _fit_pretrain(model, X_tr, Y_tr, X_val, Y_val, params, class_weight=None, val_sample_weights=None):
	model.fit(
		X_tr, Y_tr, batch_size=params['batchsize'], epochs=params['epochs'],
		shuffle=True, verbose=2,
		validation_data=(X_val, Y_val) if val_sample_weights is None else (X_val, Y_val, val_sample_weights),
		class_weight=class_weight,
		callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
	return model


def pretrain_ref_vs_ref(X_bg, X_signal, params, args, run):
	n = len(X_bg) // 2
	X_pre = np.concatenate([X_bg[:n], X_bg[n:2*n]])
	Y_pre = np.eye(2)[np.concatenate([np.ones(n), np.zeros(n)]).astype(int)]
	np.random.seed(run)
	idx = np.random.permutation(len(X_pre))
	X_pre, Y_pre = X_pre[idx], Y_pre[idx]
	model = _make_pretrain_model(params, args)
	return _fit_pretrain(model, X_pre[:n], Y_pre[:n], X_pre[n:], Y_pre[n:], params)


def pretrain_supervised(X_bg, X_signal, params, args, run):
	if len(X_signal) == 0:
		raise ValueError("pretrain_supervised: no true signal events found in training set")
	X_pre = np.concatenate([X_signal, X_bg])
	Y_pre = np.eye(2)[np.concatenate([np.ones(len(X_signal)), np.zeros(len(X_bg))]).astype(int)]
	np.random.seed(run)
	idx = np.random.permutation(len(X_pre))
	X_pre, Y_pre = X_pre[idx], Y_pre[idx]
	n = len(X_pre) // 2
	X_tr, X_val = X_pre[:n], X_pre[n:]
	Y_tr, Y_val = Y_pre[:n], Y_pre[n:]
	class_weight = {0: 1, 1: len(Y_tr) / sum(Y_tr[:, 1]) - 1}
	val_weight   = {0: 1, 1: len(Y_val) / sum(Y_val[:, 1]) - 1}
	val_sample_weights = val_weight[0] * Y_val[:, 0] + val_weight[1] * Y_val[:, 1]
	model = _make_pretrain_model(params, args)
	return _fit_pretrain(model, X_tr, Y_tr, X_val, Y_val, params, class_weight, val_sample_weights)


def make_autoencoder(inputs, params):
	hidden = int(params['hidden'])
	activation = params['activation']
	dropout = float(params['dropout'])
	l1, l2 = float(params['l1']), float(params['l2'])
	lr = float(params['lr'])
	bottleneck_dim = int(params.get('bottleneck', max(2, inputs // 4)))

	def add_activation(model):
		if activation == "relu":
			model.add(layers.ReLU())
		elif activation == "leaky":
			model.add(layers.LeakyReLU(alpha=0.1))

	model = models.Sequential()
	# Encoder — identical structure to the classifier's hidden layers
	model.add(layers.Dense(64, input_shape=(inputs,)))
	for _ in range(hidden - 1):
		add_activation(model)
		model.add(layers.Dropout(dropout))
		model.add(layers.Dense(64, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
	# Bottleneck — forces compression; encoder Dense layers stay at indices 0, 3, 6, ... (unchanged)
	add_activation(model)
	model.add(layers.Dense(bottleneck_dim))
	# Decoder
	model.add(layers.Dense(64))
	for _ in range(hidden - 1):
		add_activation(model)
		model.add(layers.Dropout(dropout))
		model.add(layers.Dense(64, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
	model.add(layers.Dense(inputs))  # linear output, MSE loss

	model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr))
	return model


def pretrain_autoencoder(X_bg, _X_signal, params, args, run):
	ae = make_autoencoder(args.inputs, params)
	np.random.seed(run)
	idx = np.random.permutation(len(X_bg))
	X_bg = X_bg[idx]
	n = len(X_bg) // 2
	X_tr, X_val = X_bg[:n], X_bg[n:]
	ae.fit(
		X_tr, X_tr, batch_size=params['batchsize'], epochs=params['epochs'],
		shuffle=True, verbose=2,
		validation_data=(X_val, X_val),
		callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

	# Transfer encoder Dense layers to a fresh classifier.
	# Encoder Dense layers are at indices 0, 3, 6, ..., 3*(hidden-1) — same positions as in make_model.
	hidden = int(params['hidden'])
	classifier = _make_pretrain_model(params, args)
	for i in range(hidden):
		classifier.layers[3 * i].set_weights(ae.layers[3 * i].get_weights())
	return classifier


def nt_xent_loss(z1, z2, temperature=0.5):
	"""NT-Xent contrastive loss. z1[i] and z2[i] are positive pairs."""
	batch_size = tf.shape(z1)[0]
	z = tf.concat([z1, z2], axis=0)                          # (2B, proj_dim)
	sim = tf.matmul(z, z, transpose_b=True) / temperature    # (2B, 2B)
	sim = sim - tf.eye(2 * batch_size) * 1e9                 # mask self-similarity
	labels = tf.concat([tf.range(batch_size, 2 * batch_size),
	                    tf.range(batch_size)], axis=0)        # positives are offset by B
	return tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim))


def pretrain_scarf_ssl(X_bg, _X_signal, params, args, run):
	hidden     = int(params['hidden'])
	act        = params['activation']
	dropout    = float(params['dropout'])
	l1_val     = float(params['l1'])
	l2_val     = float(params['l2'])
	ssl_lr     = float(params.get('scarf_ssl_lr',      1e-3))
	ssl_epochs = int(params.get('scarf_ssl_epochs',    50))
	ssl_temp   = float(params.get('scarf_ssl_temp',    0.5))
	proj_dim   = int(params.get('scarf_ssl_proj_dim',  64))
	ssl_rate   = float(params.get('scarf_ssl_rate',    0.5))
	batch_size = int(params['batchsize'])

	np.random.seed(run)
	idx = np.random.permutation(len(X_bg))
	X_tr, X_val = X_bg[idx[:len(X_bg)//2]], X_bg[idx[len(X_bg)//2:]]

	# Build encoder (mirrors classifier hidden layers; track Dense layers for weight transfer)
	inp = keras.Input(shape=(args.inputs,))
	dense_layers = []

	d = layers.Dense(64)
	dense_layers.append(d)
	h = d(inp)
	for _ in range(hidden - 1):
		h = layers.ReLU()(h) if act == 'relu' else layers.LeakyReLU(0.1)(h)
		h = layers.Dropout(dropout)(h)
		d = layers.Dense(64, kernel_regularizer=regularizers.l1_l2(l1=l1_val, l2=l2_val))
		dense_layers.append(d)
		h = d(h)
	h = layers.ReLU()(h) if act == 'relu' else layers.LeakyReLU(0.1)(h)

	# Projection head — discarded after pretraining
	proj_norm = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(layers.Dense(proj_dim)(h))
	encoder = keras.Model(inputs=inp, outputs=proj_norm)
	optimizer = keras.optimizers.Adam(ssl_lr)

	def scarf_corrupt(x):
		mask = tf.cast(tf.random.uniform(tf.shape(x)) < ssl_rate, x.dtype)
		shuffled = tf.transpose(tf.map_fn(tf.random.shuffle, tf.transpose(x)))
		return x * (1.0 - mask) + shuffled * mask

	best_val_loss, best_weights, patience_count = np.inf, encoder.get_weights(), 0
	for epoch in range(ssl_epochs):
		perm = np.random.permutation(len(X_tr))
		train_loss, n_tr = 0.0, 0
		for i in range(0, len(X_tr), batch_size):
			xb = tf.constant(X_tr[perm[i:i+batch_size]], dtype=tf.float32)
			with tf.GradientTape() as tape:
				loss = nt_xent_loss(encoder(xb, training=True),
				                    encoder(scarf_corrupt(xb), training=True), ssl_temp)
			optimizer.apply_gradients(zip(tape.gradient(loss, encoder.trainable_variables),
			                              encoder.trainable_variables))
			train_loss += loss.numpy(); n_tr += 1

		val_loss, n_val = 0.0, 0
		for i in range(0, len(X_val), batch_size):
			xb = tf.constant(X_val[i:i+batch_size], dtype=tf.float32)
			val_loss += nt_xent_loss(encoder(xb, training=False),
			                         encoder(scarf_corrupt(xb), training=False), ssl_temp).numpy()
			n_val += 1
		val_loss /= n_val

		if val_loss < best_val_loss:
			best_val_loss, best_weights, patience_count = val_loss, encoder.get_weights(), 0
		else:
			patience_count += 1
			if patience_count >= 10:
				print(f"SCARF SSL early stopping at epoch {epoch+1}")
				break
		if (epoch + 1) % 10 == 0:
			print(f"SCARF SSL epoch {epoch+1}: train={train_loss/n_tr:.4f}  val={val_loss:.4f}")

	encoder.set_weights(best_weights)

	# Transfer encoder Dense weights to classifier (by matching 64-unit Dense layers in order)
	classifier = _make_pretrain_model(params, args)
	clf_dense = [l for l in classifier.layers if isinstance(l, layers.Dense) and l.units == 64]
	for i, src in enumerate(dense_layers):
		clf_dense[i].set_weights(src.get_weights())
	return classifier


def pretrain_masked_feature(X_bg, _X_signal, params, args, run):
	hidden     = int(params['hidden'])
	act        = params['activation']
	dropout    = float(params['dropout'])
	l1_val     = float(params['l1'])
	l2_val     = float(params['l2'])
	mfp_lr     = float(params.get('mfp_lr',     1e-3))
	mfp_epochs = int(params.get('mfp_epochs',   50))
	mask_rate  = float(params.get('mask_rate',   0.3))
	batch_size = int(params['batchsize'])

	np.random.seed(run)
	idx = np.random.permutation(len(X_bg))
	X_tr, X_val = X_bg[idx[:len(X_bg)//2]], X_bg[idx[len(X_bg)//2:]]

	# Build encoder + per-feature prediction head (tracks Dense layers for weight transfer)
	inp = keras.Input(shape=(args.inputs,))
	dense_layers = []

	d = layers.Dense(64)
	dense_layers.append(d)
	h = d(inp)
	for _ in range(hidden - 1):
		h = layers.ReLU()(h) if act == 'relu' else layers.LeakyReLU(0.1)(h)
		h = layers.Dropout(dropout)(h)
		d = layers.Dense(64, kernel_regularizer=regularizers.l1_l2(l1=l1_val, l2=l2_val))
		dense_layers.append(d)
		h = d(h)
	h = layers.ReLU()(h) if act == 'relu' else layers.LeakyReLU(0.1)(h)
	pred = layers.Dense(args.inputs)(h)   # predict all features; loss only on masked ones
	model = keras.Model(inputs=inp, outputs=pred)
	optimizer = keras.optimizers.Adam(mfp_lr)

	best_val_loss, best_weights, patience_count = np.inf, model.get_weights(), 0
	for epoch in range(mfp_epochs):
		perm = np.random.permutation(len(X_tr))
		train_loss, n_tr = 0.0, 0
		for i in range(0, len(X_tr), batch_size):
			xb   = tf.constant(X_tr[perm[i:i+batch_size]], dtype=tf.float32)
			mask = tf.cast(tf.random.uniform(tf.shape(xb)) < mask_rate, tf.float32)
			with tf.GradientTape() as tape:
				loss = tf.reduce_mean(
					tf.reduce_sum(mask * tf.square(model(xb * (1.0 - mask), training=True) - xb), axis=1))
			optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
			                              model.trainable_variables))
			train_loss += loss.numpy(); n_tr += 1

		val_loss, n_val = 0.0, 0
		for i in range(0, len(X_val), batch_size):
			xb   = tf.constant(X_val[i:i+batch_size], dtype=tf.float32)
			mask = tf.cast(tf.random.uniform(tf.shape(xb)) < mask_rate, tf.float32)
			val_loss += tf.reduce_mean(
				tf.reduce_sum(mask * tf.square(model(xb * (1.0 - mask), training=False) - xb), axis=1)).numpy()
			n_val += 1
		val_loss /= n_val

		if val_loss < best_val_loss:
			best_val_loss, best_weights, patience_count = val_loss, model.get_weights(), 0
		else:
			patience_count += 1
			if patience_count >= 10:
				print(f"MFP early stopping at epoch {epoch+1}")
				break
		if (epoch + 1) % 10 == 0:
			print(f"MFP epoch {epoch+1}: train={train_loss/n_tr:.4f}  val={val_loss:.4f}")

	model.set_weights(best_weights)

	# Transfer encoder Dense weights to classifier
	classifier = _make_pretrain_model(params, args)
	clf_dense = [l for l in classifier.layers if isinstance(l, layers.Dense) and l.units == 64]
	for i, src in enumerate(dense_layers):
		clf_dense[i].set_weights(src.get_weights())
	return classifier


PRETRAIN_STRATEGIES = {
	'ref_vs_ref':     pretrain_ref_vs_ref,
	'supervised':     pretrain_supervised,
	'autoencoder':    pretrain_autoencoder,
	'scarf_ssl':      pretrain_scarf_ssl,
	'masked_feature': pretrain_masked_feature,
}


def classifier_training_pretrained(X_train, Y_train, X_test, Y_test, args, run, direc_run=None, ft_lr=None):
	if direc_run is None:
		direc_run = args.directory
	if not os.path.exists(direc_run):
		os.makedirs(direc_run)

	with open(args.cl_filename, 'r') as stream:
		params = yaml.safe_load(stream)
	if getattr(args, 'bottleneck', None) is not None:
		params['bottleneck'] = args.bottleneck

	if args.pretrain_strategy not in PRETRAIN_STRATEGIES:
		raise ValueError(f"Unknown pretrain_strategy '{args.pretrain_strategy}'. Choose from: {list(PRETRAIN_STRATEGIES)}")

	Y_true = np.load(args.directory + "Y_train_true.npy")
	X_bg = X_train[Y_train[:, 1] == 0]
	X_signal = X_train[Y_true == 1]

	strategy_fn = PRETRAIN_STRATEGIES[args.pretrain_strategy]
	model = strategy_fn(X_bg, X_signal, params, args, run)
	np.save(direc_run + 'pretrain_preds.npy', model.predict(X_test, verbose=0)[:, 1])

	if ft_lr is not None:
		model.compile(
			loss=keras.losses.CategoricalCrossentropy(label_smoothing=float(params['label_smoothing'])),
			optimizer=keras.optimizers.Adam(ft_lr, beta_1=float(params['beta_1'])),
			metrics=["accuracy"], weighted_metrics=[])

	np.random.seed(run)
	X_tr, X_val = np.array_split(X_train, 2)
	Y_tr, Y_val = np.array_split(Y_train, 2)
	class_weight = {0: 1, 1: len(Y_tr) / sum(Y_tr.T[1]) - 1}
	val_weight   = {0: 1, 1: len(Y_val) / sum(Y_val.T[1]) - 1}
	val_sample_weights = val_weight[0] * Y_val[:, 0] + val_weight[1] * Y_val[:, 1]

	results = model.fit(X_tr, Y_tr, batch_size=params['batchsize'], epochs=params['epochs'],
		shuffle=True, verbose=2, validation_data=(X_val, Y_val, val_sample_weights),
		class_weight=class_weight,
		callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

	np.save(direc_run + 'classifier_history.npy', results.history)
	test_results = model.predict(X_test, verbose=0).T[1]
	print("AUC with averaging: %.3f" % pf.plot_roc(test_results, Y_test[:, 1], title="roc_NN", directory=args.directory, direc_run=direc_run))
	np.save(direc_run + "preds.npy", test_results)
	return model, results


def classifier_training(X_train, Y_train, X_test, Y_test, args, run, direc_run=None):
	if direc_run is None:
		direc_run=args.directory

	with open(args.cl_filename, 'r') as stream:
		params = yaml.safe_load(stream)

	model = make_model(activation=params['activation'], hidden=int(params['hidden']), inputs=args.inputs, lr=float(params['lr']), dropout=float(params['dropout']), l1=float(params['l1']), l2=float(params['l2']), momentum=float(params['beta_1']), label_smoothing=float(params['label_smoothing']))

	if not os.path.exists(direc_run):
		os.makedirs(direc_run)

	earlystopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
	callbacks = [earlystopping]

	np.random.seed(run)
	inds = np.array(range(len(X_train)))
	np.random.shuffle(inds)
	X_train, X_val = np.array_split(X_train,2)
	Y_train, Y_val = np.array_split(Y_train,2)

	class_weight = {0: 1, 1: len(Y_train)/sum(Y_train.T[1])-1}
	val_weight = {0: 1, 1: len(Y_val)/sum(Y_val.T[1])-1}
	val_sample_weights = val_weight[0]*Y_val[:,0]+val_weight[1]*Y_val[:,1]

	results = model.fit(
		X_train,
		Y_train,
		batch_size=params['batchsize'],
		epochs=params['epochs'],
		shuffle=True,
		verbose=2,
		validation_data=(X_val, Y_val, val_sample_weights),
		class_weight=class_weight,
		callbacks=callbacks,
	)

	np.save(direc_run+'classifier_history.npy', results.history)

	test_results = model.predict(X_test, verbose=0).T[1]
	print("AUC with averaging: %.3f" % pf.plot_roc(test_results, Y_test[:,1], title="roc_NN",directory=args.directory, direc_run=direc_run))
	np.save(direc_run+"preds.npy", test_results)

	return model, results
