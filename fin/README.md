### CNN 메모
#### CNN 특성 추출기로만 사용되었을 경우
* compile, fit 대신 바로 predict를 호출하여 특성을 추출함 
* Flatten() 레이어를 통해 출력된 벡터가 특성 벡터로 사용됨 
```text
ValueError: Arguments target and output must have the same shape. Received: target.shape=(None, 5), output.shape=(None, 4096)
```
* CNN은 특성 추출기로 설계되었지만 (특성 백터 생성) 분류기로 사용되었을 경우 (라벨 반환)
* CNN은 이미지의 고차원 형태를 고차원 특성을 백터 형태로 추출하고, 추출된 특성을 외부 분류기에 전달하여 예측을 수행하는 것 

#### CNN이 분류기로 사용되는 경우
  * 마지막에 Dense 레이어를 추가하고, 출력 형태와 타겟 라벨을 일치시켜야 함 (sigmoid 함수 사용)
 

#### CNN 특성 추출기로만 사용되었을 때 원핫 인코딩
  * CNN 모델의 역할은 특징 백터를 생성하는 것 -> 레이블 정보가 포함되지 않음
  * 레이블의 정보는 CNN을 학습시키거나, cNN이 분류기로 사용되었을 경우만 필요함
  * 원핫 인코딩을 해도, 안 해도 결과는 동일했음 -> CNN이 특성 추출기로만 사용된 모든 코드에서 원핫 인코딩을 제거하기로 결정 (레이블 데이터 정수형으로 유지하기)
 *  아래 `np.argmax`는 원핫인코딩을 하지 않으므로 고려해도 되지 않음 (원핫인코딩을 진행하여 생각한 것)
----
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


