### knn 간단 메모

cnn에서

```python
keras.layers.Flatten()
```

위 코드를 추가하지 않으면


```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-5-c87b139da04b> in <cell line: 83>()
     81
     82 knn = KNeighborsClassifier(n_neighbors=5)
---> 83 knn.fit(X_train_features, np.argmax(train_labels, axis=1))  # One-hot -> integer labels
     84
     85 # KNN 예측 및 평가

5 frames
/usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
   1056             )
   1057         if not allow_nd and array.ndim >= 3:
-> 1058             raise ValueError(
   1059                 "Found array with dim %d. %s expected <= 2."
   1060                 % (array.ndim, estimator_name)

ValueError: Found array with dim 4. KNeighborsClassifier expected <= 2.
```

array가 4차원이고, KNeighborsClassifier는 2차원 이하의 데이터만 처리할 수 있어 에러가 발생한다. 따라서 이를 모두 1열로 펼치는 Flatten()을 이용했다.

```
X_train_features = cnn_model.predict(train_images)
X_test_features = cnn_model.predict(test_images)
```
이미지 데이터에서 특성 백터만을 추출한다. 이는 KNN의 입력 데이터로 들어간다. (MLP 연결 x)

```
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_features, np.argmax(train_labels, axis=1))
```
이때, fit을 할 때 데이터와 레이블의 개수가 모두 동일해야 한다.
train_labels는 원핫 인코딩되어있기 때문에 2의 경우 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 형태의 데이터가 들어가있다.
np.argmax는 axis=1 배열을 1d로 간주했을 때 최대값의 위치를 반환한다. 따라서 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]는 label 2로 간주되며, 이었다면 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 이 모양이 되므로 label 0으로 간주된다.
따라서 클래스를 분류하는 정수 레이블로 대체할 수 있다. (맞나?)

이후 이 데이터를 가지고 fit 메서드를 사용하여 knn에 입력한다.


