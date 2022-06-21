import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Вычислите метрики для бинарной классификации

    Аргументы:
    prediction, np array of bool (num_samples) - предсказания модели
    ground_truth, np array of bool (num_samples) - эталонные значения

    Возвращает:
    precision, recall, f1, accuracy - метрики классификации
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: реализуйте метрики!
    # Полезные источники:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp = np.count_nonzero(np.logical_and(prediction, ground_truth))
    fp = np.count_nonzero(np.logical_and(prediction, np.logical_not(ground_truth)))
    fn = np.count_nonzero(np.logical_and(np.logical_not(prediction), ground_truth))
    tn = np.count_nonzero(np.logical_and(np.logical_not(prediction), np.logical_not(ground_truth)))
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    if (tp + tn + fp + fn) != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    if (fp + fn) != 0:
        f1 = tp / (tp + 0.5 * (fp + fn))
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Вычислите метрики для мультиклассовой классификации

    Аргументы:
    prediction, np array of int (num_samples) - предсказания модели
    ground_truth, np array of int (num_samples) - эталонные значения

    Возвращает:
    accuracy - точность предсказаний

    tp = np.zeros((np.amax(ground_truth) + 1), dtype=int)
    fn = np.zeros((np.amax(ground_truth) + 1), dtype=int)
    tn = np.zeros((np.amax(ground_truth) + 1), dtype=int)
    accuracy = 0
    # TODO: Реализуйте вычисление точности
    for pr, gt in zip(prediction, ground_truth):
        if pr == gt:
            tp[pr]+=1
        else:
            fn[pr]+=1
    tn = ground_truth.size - tp - 2 * fn
    sum1=np.sum(tn, dtype = np.int64)
    sum2=np.sum(tp, dtype = np.int64)
    sum3=2*np.sum(fn, dtype = np.int64)+sum1+sum2
    accuracy=(sum1+sum2)/sum3
    '''

    correct_predictions = 0
    for pr, gt in zip(prediction, ground_truth):
        if pr == gt:
            correct_predictions += 1
    accuracy = correct_predictions/len(ground_truth)
    return accuracy
