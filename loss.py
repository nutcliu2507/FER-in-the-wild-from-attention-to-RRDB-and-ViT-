import tensorflow.keras.backend as K
def CE(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    CE_loss = - y_true[...] * K.log(y_pred)
    CE_loss = alpha * K.pow(1 - y_pred, gamma) * CE_loss
    CE_loss = K.mean(K.sum(CE_loss, axis=-1))
    return CE_loss


